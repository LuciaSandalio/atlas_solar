import markdown

# Read the markdown file
with open('/home/chula/.gemini/antigravity/brain/073532f8-c3dc-4505-bf93-5b7b0f33c6a1/reporte_fase4_tercer_informe.md', 'r') as f:
    text = f.read()

# Replace the absolute image path with the relative path or absolute path so browser can read it
text = text.replace('/home/chula/.gemini/antigravity/brain/073532f8-c3dc-4505-bf93-5b7b0f33c6a1/fase4_seasonal_rmse_comparison.png', 'fase4_seasonal_rmse_comparison.png')

# Convert Markdown to HTML
html_content = markdown.markdown(text, extensions=['tables'])

# Add some basic CSS for a beautiful PDF
css_content = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 850px;
    margin: 0 auto;
    padding: 40px;
}
h1, h2, h3 {
    color: #2c3e50;
    margin-top: 30px;
    margin-bottom: 15px;
    font-weight: 600;
}
h1 { font-size: 28px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h2 { font-size: 22px; border-bottom: 1px solid #eee; padding-bottom: 8px; }
p { margin-top: 0; margin-bottom: 16px; }
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 25px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
th, td {
    padding: 12px 15px;
    border: 1px solid #ddd;
    text-align: center;
}
th:first-child, td:first-child { text-align: left; }
th {
    background-color: #f8f9fa;
    font-weight: 600;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 30px auto;
    border-radius: 4px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
blockquote {
    border-left: 4px solid #3498db;
    padding: 15px 20px;
    margin: 20px 0;
    background-color: #f8f9fa;
    color: #555;
    font-style: italic;
    border-radius: 0 4px 4px 0;
}
@media print {
    body { padding: 0; }
    table { box-shadow: none; border: 1px solid #ddd; }
    img { box-shadow: none; }
}
"""

# Create the final HTML string
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Reporte Fase 4</title>
    <style>{css_content}</style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Output HTML
with open('results/reporte_fase4.html', 'w') as f:
    f.write(full_html)
print("HTML generated successfully: results/reporte_fase4.html")
