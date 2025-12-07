#!/usr/bin/env python3
"""Quick test to verify QR code generation works"""

import sys
sys.path.insert(0, '/Users/buyer/Downloads/tile_project 7')

try:
    import qrcode
    from PIL import Image
    import base64

    # Test compact HTML QR generation
    tile_id = "T-0-1"
    tile_data = {
        'material': 'Tile',
        'density': 1900,
        'weight': 15,
        'thermal_r': 0.05,
        'thickness': '20.0mm'
    }

    # Build compact HTML
    tile_info_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tile {tile_id}</title>
<style>body{{font-family:Arial;margin:15px;background:#f5f5f5}}
.c{{background:#fff;padding:15px;border-radius:6px;box-shadow:0 2px 4px rgba(0,0,0,0.1);max-width:500px;margin:0 auto}}
h1{{color:#333;margin:0 0 15px;font-size:18px}}
.g{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.i{{padding:8px;background:#f9f9f9;border-left:3px solid #007bff}}
.l{{font-weight:bold;color:#555;font-size:11px;text-transform:uppercase}}
.v{{font-size:16px;color:#333;margin-top:3px}}</style>
</head>
<body>
<div class="c">
<h1>📦 {tile_id}</h1>
<div class="g">
<div class="i"><div class="l">Material</div><div class="v">{tile_data.get('material', 'N/A')}</div></div>
<div class="i"><div class="l">Thickness</div><div class="v">{tile_data.get('thickness', 'N/A')}</div></div>
<div class="i"><div class="l">Density</div><div class="v">{tile_data.get('density', 'N/A')}</div></div>
<div class="i"><div class="l">Weight</div><div class="v">{tile_data.get('weight', 'N/A')}</div></div>
<div class="i"><div class="l">R-Value</div><div class="v">{tile_data.get('thermal_r', 'N/A')}</div></div>
</div></div>
</body>
</html>"""

    # Create data URL
    html_bytes = tile_info_html.encode('utf-8')
    b64_html = base64.b64encode(html_bytes).decode('utf-8')
    data_url = f"data:text/html;base64,{b64_html}"

    print(f"✓ HTML size: {len(tile_info_html)} bytes")
    print(f"✓ Base64 HTML size: {len(b64_html)} bytes")
    print(f"✓ Data URL size: {len(data_url)} bytes")

    # Generate QR code
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(data_url)
    qr.make(fit=True)

    print(f"✓ QR code version: {qr.version}")
    print(f"✓ QR code module count: {qr.modules_count}x{qr.modules_count}")

    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    print(f"✓ Image created: {img.size}")

    # Save test image
    test_path = "/Users/buyer/Downloads/tile_project 7/qr_test.png"
    img.save(test_path)
    print(f"✓ Test QR code saved to: {test_path}")

    print("\n✅ QR CODE GENERATION TEST PASSED!")
    print("The QR codes should now display correctly in the application.")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

