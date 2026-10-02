import markdown
from weasyprint import HTML, CSS

# Read the markdown file
with open('/home/chula/.gemini/antigravity/brain/073532f8-c3dc-4505-bf93-5b7b0f33c6a1/reporte_fase4_tercer_informe.md', 'r') as f:
    text = f.read()

# Convert Markdown to HTML
html_content = markdown.markdown(text, extensions=['tables'])

# Add some basic CSS for a beautiful PDF
css_content = """
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
h1, h2, h3 {
    color: #2c3e50;
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
}
h1 { font-size: 24px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
h2 { font-size: 20px; border-bottom: 1px solid #eee; padding-bottom: 8px; }
p { margin-top: 0; margin-bottom: 16px; }
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 16px;
}
th, td {
    padding: 8px 12px;
    border: 1px solid #ddd;
}
th {
    background-color: #f8f9fa;
    font-weight: 600;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
}
blockquote {
    border-left: 4px solid #3498db;
    padding: 10px 15px;
    margin: 0 0 20px;
    background-color: #f8f9fa;
    color: #555;
    font-style: italic;
}
"""

# Create the final HTML string
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>{css_content}</style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Generate PDF
HTML(string=full_html, base_url='/home/chula/Escritorio/atlas_solar/').write_pdf('resultados_fase4_reporte.pdf')
print("PDF generated successfully: resultados_fase4_reporte.pdf")
