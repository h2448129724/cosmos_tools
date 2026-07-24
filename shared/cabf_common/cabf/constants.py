# Canonical image-format universe. IMAGE_SUFFIX_ORDER is the deterministic
# enumeration order (most-common first); IMAGE_SUFFIXES is the derived
# membership set. Consumers that scan for an image by trying extensions should
# iterate IMAGE_SUFFIX_ORDER; consumers that only test "is this an image" use
# IMAGE_SUFFIXES. Defining both here is the single source of truth so the two
# can never drift (the historical webp-discovery bug came from such drift).
IMAGE_SUFFIX_ORDER = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
IMAGE_SUFFIXES = set(IMAGE_SUFFIX_ORDER)
MASTER_SCHEMA_VERSION = "1.2"
POINT_LABEL_ALIASES = {"sew", "keypoint"}
