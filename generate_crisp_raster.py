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
    # 1. Download a crisp 512px raster version from Google Favicon service
    url = "https://t3.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.formula1.com&size=512"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    img_data = urllib.request.urlopen(req).read()

    # 2. Open this high-res image
    img = Image.open(io.BytesIO(img_data)).convert("RGBA")

    # 3. Create a solid 512x512 background (dark racing color: #15151e)
    bg = Image.new('RGBA', (512, 512), (21, 21, 30, 255))
    
    # 4. Paste the F1 logo over the background
    try:
        bg.paste(img, (0, 0), img)
    except ValueError:
        bg.paste(img, (0, 0))

    # 5. Save the 180x180 version for Apple touch icon
    apple_icon = bg.resize((180, 180), Image.Resampling.LANCZOS).convert('RGB')
    apple_icon.save("apple-touch-icon.png")
    
    # Save a 512x512 version for the modern manifest
    manifest_icon = bg.convert('RGB')
    manifest_icon.save("icon-512.png")
    
    print("Successfully generated high-res apple-touch-icon.png and icon-512.png")
except Exception as e:
    print(f"Error: {e}")
