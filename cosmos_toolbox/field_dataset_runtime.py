"""Isolated execution; file-based cooperative control and structured stdout events."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
import time
from types import SimpleNamespace

PREFIX = 'FIELD_DATASET_EVENT '


def run_in_environment(options, environment_name, control, on_progress, scan_only=False):
    from .paths import ensure_import_paths
    ensure_import_paths()
    from shared.conda_runtime import CondaEnvManager
    manager = CondaEnvManager()
    selected = manager.find(environment_name)
    if selected is None or not Path(selected.python_executable).is_file():
        raise ValueError(f'Conda 环境不存在或没有 Python：{environment_name}')
    environment = manager.process_environment(environment_name)
    environment['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parents[1]),
                                               environment.get('PYTHONPATH', '')])
    environment['PYTHONIOENCODING'] = 'utf-8'
    environment['PYTHONUNBUFFERED'] = '1'
    environment.pop('PYTHONHOME', None)
    on_progress({'message': f'执行环境：{selected.name} | {selected.python_executable}'})
    process = subprocess.Popen([selected.python_executable, '-u', '-m',
                                'cosmos_toolbox.field_dataset_runtime'],
        cwd=str(Path(__file__).resolve().parents[3]), env=environment,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', bufsize=1,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    finished = threading.Event()
    mailbox = tempfile.TemporaryDirectory(prefix='field-dataset-control-')
    control_path = Path(mailbox.name) / 'control.json'

    def send(value):
        process.stdin.write(json.dumps(value, ensure_ascii=False) + '\n')
        process.stdin.flush()

    def controls():
        while not finished.wait(.1):
            state = (control.paused.is_set(), control.stopped.is_set())
            try:
                temporary = control_path.with_suffix('.tmp')
                temporary.write_text(json.dumps({'paused': state[0], 'stopped': state[1]}), encoding='utf-8')
                os.replace(temporary, control_path)
            except OSError:
                continue

    result = None
    failure = None
    thread = None
    try:
        send({'options': options, 'scan_only': scan_only, 'control_path': str(control_path)})
        thread = threading.Thread(target=controls, daemon=True)
        thread.start()
        for line in process.stdout:
            if not line.startswith(PREFIX):
                if line.strip():
                    on_progress({'message': line.rstrip()})
                continue
            event = json.loads(line[len(PREFIX):])
            if event['type'] == 'result':
                result = event['data']
            elif event['type'] == 'error':
                failure = event['data']
            else:
                on_progress(event['data'])
        code = process.wait()
        if failure or code or result is None:
            raise RuntimeError(failure or f'环境 {environment_name} 的任务进程异常退出（{code}），请查看日志')
        return result
    finally:
        finished.set()
        if thread:
            thread.join(timeout=1)
        if process.poll() is None:
            process.terminate()
            process.wait()
        process.stdin.close()
        process.stdout.close()
        mailbox.cleanup()


def main():
    def emit(kind, data):
        print(PREFIX + json.dumps({'type': kind, 'data': data}, ensure_ascii=False), flush=True)

    request = json.loads(sys.stdin.readline())
    control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
    control_path = Path(request['control_path'])

    def controls():
        started = time.time()
        while True:
            try:
                state = json.loads(control_path.read_text(encoding='utf-8'))
                started = time.time()
                (control.paused.set if state.get('paused') else control.paused.clear)()
                if state.get('stopped') or time.time() - control_path.stat().st_mtime > 60:
                    control.stopped.set()
                    control.paused.clear()
                    return
            except (OSError, ValueError):
                if not control_path.parent.exists() or time.time() - started > 60:
                    control.stopped.set()
                    control.paused.clear()
                    return
            time.sleep(.1)

    threading.Thread(target=controls, daemon=True).start()
    try:
        from .paths import ensure_import_paths
        ensure_import_paths()
        from .field_dataset import run, scan
        options = request['options']
        emit('progress', {'message': f'任务 Python：{sys.executable}'})
        if request.get('scan_only'):
            files = scan(options['source'], face=options['face'])
            result = {'scan': True, 'count': len(files), 'bytes': sum(p.stat().st_size for p in files)}
        else:
            result = run(**options, control=control, on_progress=lambda value: emit('progress', value))
        emit('result', result)
    except Exception as exc:
        emit('error', f'{type(exc).__name__}: {exc}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
