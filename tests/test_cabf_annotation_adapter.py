import json

from apps.labeling_ui.app.annotation.adapters.cabf import CabfAnnotationAdapter


def test_cabf_adapter_loads_points_edges_roi_and_segments(tmp_path):
    image_path = tmp_path / "sample.png"
    annotation_path = tmp_path / "sample.json"
    annotation_path.write_text(
        json.dumps(
            {
                "schema_version": "1.2",
                "sample_id": "sample",
                "image_path": "sample.png",
                "image_size": {"width": 640, "height": 480},
                "roi": [10, 20, 110, 120],
                "points": [
                    {"id": 0, "x": 12.5, "y": 20.0, "score": 0.9, "source": "model"},
                    {"id": 1, "x": 40.0, "y": 80.0, "score": 1.0, "source": "manual"},
                ],
                "edges": [
                    {"edge_id": "edge_0001", "src": 0, "dst": 1, "label": 1, "source": "manual"}
                ],
                "segments": [
                    {"id": "seg_1", "points": [[1, 2], [3, 4], [5, 2]], "label": "region"}
                ],
                "metadata": {"origin": "unit-test"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    document = CabfAnnotationAdapter().load(image_path, annotation_path)

    assert document.image_path == str(image_path)
    assert document.image_size.width == 640
    assert document.image_size.height == 480
    assert document.get_layer("points").visible is True
    assert document.get_layer("edges").visible is True
    assert document.get_layer("roi").visible is False
    assert document.get_layer("segments").visible is False
    assert document.get_layer("points").shapes[0].geometry == {"x": 12.5, "y": 20.0}
    assert document.get_layer("edges").shapes[0].geometry["refs"] == ["0", "1"]
    assert document.get_layer("roi").shapes[0].geometry == {"x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 120.0}
    assert document.metadata["cabf"]["origin"] == "unit-test"


def test_cabf_adapter_dumps_document_back_to_master_json(tmp_path):
    image_path = tmp_path / "sample.png"
    output_path = tmp_path / "out.json"
    adapter = CabfAnnotationAdapter()
    document = adapter.new_empty(image_path, width=320, height=240)
    document.get_layer("points").shapes.extend(
        [
            adapter.point_shape(point_id=0, x=10, y=20, attrs={"score": 0.8, "source": "model"}),
            adapter.point_shape(point_id=1, x=30, y=40, attrs={"score": 1.0, "source": "manual"}),
        ]
    )
    document.get_layer("edges").shapes.append(
        adapter.edge_shape(edge_id="edge_0001", src="0", dst="1", attrs={"label": 1, "source": "manual"})
    )
    document.get_layer("roi").shapes.append(
        adapter.box_shape(shape_id="roi", x1=1, y1=2, x2=100, y2=120)
    )

    adapter.dump(document, output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["sample_id"] == "sample"
    assert data["image_size"] == {"width": 320, "height": 240}
    assert data["points"][0]["id"] == 0
    assert data["points"][0]["x"] == 10.0
    assert data["edges"][0]["src"] == 0
    assert data["edges"][0]["dst"] == 1
    assert data["roi"] == [1.0, 2.0, 100.0, 120.0]
    assert data["metadata"]["source"] == "generic_annotation_adapter"


def test_cabf_adapter_converts_document_to_master_dict(tmp_path):
    image_path = tmp_path / "sample.png"
    adapter = CabfAnnotationAdapter()
    document = adapter.new_empty(image_path, width=64, height=32)
    document.get_layer("points").shapes.append(
        adapter.point_shape(point_id=0, x=5, y=6, attrs={"source": "manual"})
    )

    data = adapter.to_master_dict(document)

    assert data["sample_id"] == "sample"
    assert data["image_path"] == str(image_path)
    assert data["image_size"] == {"width": 64, "height": 32}
    assert data["points"] == [{"id": 0, "x": 5.0, "y": 6.0, "score": 1.0, "source": "manual"}]


def test_cabf_adapter_builds_document_from_master_dict(tmp_path):
    image_path = tmp_path / "sample.png"
    adapter = CabfAnnotationAdapter()

    document = adapter.from_master_dict(
        {
            "sample_id": "sample",
            "image_path": "sample.png",
            "image_size": {"width": 50, "height": 40},
            "points": [{"id": 0, "x": 7, "y": 8}],
            "edges": [],
        },
        image_path=image_path,
    )

    assert document.image_path == str(image_path)
    assert document.image_size.width == 50
    assert document.image_size.height == 40
    assert document.get_layer("points").shapes[0].geometry == {"x": 7.0, "y": 8.0}
