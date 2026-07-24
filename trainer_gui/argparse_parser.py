from __future__ import annotations

import ast
from typing import Any

from .models import FeatureAction, FieldSchema


class ArgparseSchemaParser:
    PATH_HINT_NAMES = {"img_dir", "ann_dir", "save_dir", "output_dir"}
    PATH_HINT_SUFFIXES = ("_dir", "_path")

    def parse_action(self, action: FeatureAction) -> list[FieldSchema]:
        source = action.script_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(action.script_path))
        constants = self._collect_constants(tree)
        fields: list[FieldSchema] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not self._is_add_argument_call(node):
                continue
            field = self._parse_add_argument(node, constants)
            if field is None:
                continue
            fields.append(field)

        fields = self._deduplicate(fields)
        fields = self._apply_action_rules(action, fields)
        action.schema = fields
        return fields

    @staticmethod
    def _is_add_argument_call(node: ast.Call) -> bool:
        return isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"

    def _parse_add_argument(self, node: ast.Call, constants: dict[str, Any]) -> FieldSchema | None:
        if not node.args:
            return None
        first_arg = self._literal_value(node.args[0], constants)
        if not isinstance(first_arg, str) or not first_arg.startswith("--"):
            return None

        cli_flag = first_arg
        name = cli_flag.lstrip("-").replace("-", "_")
        field_type = "string"
        required = False
        default: Any = None
        help_text = ""
        choices: list[Any] | None = None
        action: str | None = None

        for keyword in node.keywords:
            if keyword.arg == "required":
                required = bool(self._literal_value(keyword.value, constants))
            elif keyword.arg == "default":
                default = self._literal_value(keyword.value, constants, fallback=ast.unparse(keyword.value))
            elif keyword.arg == "help":
                help_text = str(self._literal_value(keyword.value, constants, fallback="") or "")
            elif keyword.arg == "choices":
                raw_choices = self._literal_value(keyword.value, constants)
                if isinstance(raw_choices, (list, tuple)):
                    choices = list(raw_choices)
            elif keyword.arg == "action":
                action = str(self._literal_value(keyword.value, constants, fallback="") or "")
            elif keyword.arg == "type":
                field_type = self._infer_type(keyword.value)

        widget = self._infer_widget(name, field_type, action, choices)
        if action in {"store_true", "store_false"}:
            field_type = "bool"
            if default is None:
                default = action == "store_false"

        return FieldSchema(
            name=name,
            cli_flag=cli_flag,
            field_type=field_type,
            required=required,
            default=default,
            help=help_text,
            choices=choices,
            action=action,
            widget=widget,
            path_mode=None,
            group=None,
            label=name,
        )

    @staticmethod
    def _collect_constants(tree: ast.AST) -> dict[str, Any]:
        constants: dict[str, Any] = {}
        for node in getattr(tree, "body", []):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            try:
                constants[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
        return constants

    @staticmethod
    def _literal_value(node: ast.AST, constants: dict[str, Any] | None = None, fallback: Any = None) -> Any:
        if isinstance(node, ast.Name) and constants and node.id in constants:
            return constants[node.id]
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            return fallback

    @staticmethod
    def _infer_type(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return {
                "str": "string",
                "int": "int",
                "float": "float",
                "bool": "bool",
            }.get(node.id, "string")
        return "string"

    def _infer_widget(
        self,
        name: str,
        field_type: str,
        action: str | None,
        choices: list[Any] | None,
    ) -> str:
        if action in {"store_true", "store_false"}:
            return "checkbox"
        if choices:
            return "select"
        if name in self.PATH_HINT_NAMES or name.endswith(self.PATH_HINT_SUFFIXES):
            return "path"
        if field_type == "int":
            return "number"
        if field_type == "float":
            return "float"
        return "text"

    @staticmethod
    def _deduplicate(fields: list[FieldSchema]) -> list[FieldSchema]:
        seen: dict[str, FieldSchema] = {}
        for field in fields:
            seen[field.name] = field
        return list(seen.values())

    def _apply_action_rules(self, action: FeatureAction, fields: list[FieldSchema]) -> list[FieldSchema]:
        hidden = set(action.ui.hidden_fields)
        read_only = set(action.ui.read_only_fields)
        ordered_names = action.ui.field_order
        overrides = action.field_overrides

        for field in fields:
            if field.name in hidden:
                field.hidden = True
            if field.name in read_only:
                field.read_only = True
            override = overrides.get(field.name)
            if not override:
                continue
            if override.label is not None:
                field.label = override.label
            if override.help is not None:
                field.help = override.help
            if override.widget is not None:
                field.widget = override.widget
            if override.path_mode is not None:
                field.path_mode = override.path_mode
            if override.group is not None:
                field.group = override.group
            if override.placeholder is not None:
                field.placeholder = override.placeholder
            if override.required is not None:
                field.required = override.required
            if override.default is not None:
                field.default = override.default
            if override.choices is not None:
                field.choices = override.choices
            if override.hidden is not None:
                field.hidden = override.hidden
            if override.read_only is not None:
                field.read_only = override.read_only

        visible = [field for field in fields if not field.hidden]
        if not ordered_names:
            return visible

        order_map = {name: idx for idx, name in enumerate(ordered_names)}
        return sorted(visible, key=lambda item: (order_map.get(item.name, 10_000), item.name))
