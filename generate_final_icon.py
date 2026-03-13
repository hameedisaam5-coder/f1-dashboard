import sys
import subprocess
try:
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image
import urllib.request
import io

try:
    # 1. Download a crisp 1024px wide version directly from Wikipedia's SVG renderer
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1024px-F1.svg.png"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'})
    img_data = urllib.request.urlopen(req).read()

    # 2. Open this high-res transparent image
    img = Image.open(io.BytesIO(img_data)).convert("RGBA")

    # 3. Create a solid 1024x1024 background (dark racing color: #15151e)
    bg = Image.new('RGBA', (1024, 1024), (21, 21, 30, 255))
    
    # 4. Center the F1 logo on the 1024x1024 background
    x_offset = (1024 - img.width) // 2
    y_offset = (1024 - img.height) // 2
    
    # Paste using the logo's alpha channel as a mask
    bg.paste(img, (x_offset, y_offset), img)

    # 5. Save the 180x180 version for Apple touch icon
    apple_icon = bg.resize((180, 180), Image.Resampling.LANCZOS).convert('RGB')
    apple_icon.save("apple-touch-icon.png")
    
    # Save a 512x512 version for the modern manifest
    manifest_icon = bg.resize((512, 512), Image.Resampling.LANCZOS).convert('RGB')
    manifest_icon.save("icon-512.png")
    
    print("Successfully generated centered high-res apple-touch-icon.png and icon-512.png")
except Exception as e:
    print(f"Error: {e}")
