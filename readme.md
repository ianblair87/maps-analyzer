#  [Maps analyzer](https://github.com/KiK0S/maps-analyzer)

[data](https://drive.google.com/drive/folders/15UIXN3eHVnJFqlWF3cBCtR7gJdv5027D?usp=sharing)


Overall idea: orienteering maps analyzer

### How to run:

```
python3 -m pip install -r requirements.txt
python3 -m streamlit run streamlit_app.py 
```

This will open website where maps can be uploaded and processed

### To contribute more data:

Labelling tool used for creating semantic segmentation dataset is [labelme](https://github.com/wkentaro/labelme)
We use a convention of class names:

| label | objects |
---
| water | lakes, rivers, etc |
| fence | fences that are prohibited to cross |
| building | buildings that you can't run into |
| forbidden | any other type of area that is forbidden to run across |
| course | course-related signs: start, controls, finish |
