Background stadium image
========================

Save your stadium photo in THIS folder with the exact name:

    emirates_stadium.jpg

Full path:
    D:\GenAI in Football\GENAI in Football\static\emirates_stadium.jpg

The app serves it at /static/emirates_stadium.jpg and uses it as the landing
background. If the file is missing, the app automatically falls back to the
built-in SVG stadium, so nothing breaks.

After adding it, commit so it deploys on Render:
    git add static/emirates_stadium.jpg
    git commit -m "Add Emirates stadium background image"
    git push
