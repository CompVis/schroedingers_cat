from pathlib import Path
import math

from PIL import Image


SOURCE = Path(
    "/Users/jannik/.codex/generated_images/019f940f-b517-7181-9a57-411d6a84d497/"
    "call_zBlrSnlPDaD7tH99z2hX0ZJV.png"
)
OUTPUT = Path("static/images/generic-orange-cat-favicon.png")
KEY_COLOR = (0, 255, 0)


def remove_chroma_key(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    pixels = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            red, green, blue, alpha = pixels[x, y]
            distance = math.sqrt(
                (red - KEY_COLOR[0]) ** 2 + (green - KEY_COLOR[1]) ** 2 + (blue - KEY_COLOR[2]) ** 2
            )
            if distance < 65:
                pixels[x, y] = (red, green, blue, 0)
            elif distance < 135:
                pixels[x, y] = (red, green, blue, int(alpha * (distance - 65) / 70))

    return image


def center_square(image: Image.Image) -> Image.Image:
    bbox = image.getbbox()
    cropped = image.crop(bbox) if bbox is not None else image
    canvas_size = max(cropped.size)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    canvas.alpha_composite(cropped, ((canvas_size - cropped.width) // 2, (canvas_size - cropped.height) // 2))
    return canvas


def main() -> None:
    icon = center_square(remove_chroma_key(Image.open(SOURCE)))
    icon = icon.resize((128, 128), Image.Resampling.LANCZOS)
    icon.save(OUTPUT, optimize=True)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
