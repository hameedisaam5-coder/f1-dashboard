import sys
import subprocess
import urllib.request

try:
    import cairosvg
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg"])
    import cairosvg

try:
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

try:
    # 1. Download the original SVG from Wikipedia
    url = "https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    svg_data = urllib.request.urlopen(req).read()

    # 2. Convert SVG directly to a large internal PNG representation (1024x1024 for crispness)
    # The original SVG might not be square, so output it temporarily
    cairosvg.svg2png(bytestring=svg_data, write_to="temp_f1.png", output_width=800)

    # 3. Open this new high-res transparent PNG
    fg = Image.open("temp_f1.png").convert("RGBA")

    # 4. Create a solid 1024x1024 background (dark racing color: #15151e)
    bg = Image.new('RGBA', (1024, 1024), (21, 21, 30, 255))
    
    # 5. Calculate position to center the F1 logo on the 1024x1024 background
    x_offset = (1024 - fg.width) // 2
    y_offset = (1024 - fg.height) // 2
    
    # Paste using the logo's alpha channel as a mask
    bg.paste(fg, (x_offset, y_offset), fg)

    # 6. Resize down to Apple's recommended highest resolution standard (180x180) using high-quality resampling
    # We will also save a 512x512 for the manifest to ensure the highest possible quality for PWA
    
    # Save the 180x180 version for backwards compatibility with link tag
    apple_icon = bg.resize((180, 180), Image.Resampling.LANCZOS).convert('RGB')
    apple_icon.save("apple-touch-icon.png")
    
    # Save a 512x512 version for the modern manifest
    manifest_icon = bg.resize((512, 512), Image.Resampling.LANCZOS).convert('RGB')
    manifest_icon.save("icon-512.png")
    
    print("Successfully generated high-res apple-touch-icon.png and icon-512.png")
except Exception as e:
    print(f"Error: {e}")
