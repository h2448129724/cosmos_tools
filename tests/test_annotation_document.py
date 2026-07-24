from apps.labeling_ui.app.annotation.document import (
    AnnotationDocument,
    AnnotationLayer,
    AnnotationShape,
    ImageSize,
    LayerStyle,
)


def test_document_finds_layers_by_key():
    document = AnnotationDocument(
        image_path="sample.png",
        image_size=ImageSize(width=100, height=80),
        layers=[
            AnnotationLayer(
                key="points",
                title="Points",
                shape_type="point",
                source_field="points",
                shapes=[
                    AnnotationShape(
                        id="0",
                        geometry={"x": 1.0, "y": 2.0},
                        attrs={"source": "manual"},
                    )
                ],
                style=LayerStyle(color="#50ff50"),
            )
        ],
    )

    layer = document.get_layer("points")

    assert layer is not None
    assert layer.key == "points"
    assert layer.shapes[0].geometry == {"x": 1.0, "y": 2.0}
    assert document.get_layer("edges") is None
