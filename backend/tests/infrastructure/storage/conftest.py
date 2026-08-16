import io

import pytest
from PIL import Image


@pytest.fixture
def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color="red").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color="blue").save(buffer, format="PNG")
    return buffer.getvalue()
