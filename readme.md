#  [Maps analyzer](https://github.com/KiK0S/maps-analyzer)

[data](https://drive.google.com/drive/folders/15UIXN3eHVnJFqlWF3cBCtR7gJdv5027D?usp=sharing)


Overall idea: to analyze maps 

### expected result:
- [ ] get data
- [ ] detect "runnable" map
- [ ] advanced: get speedmap based on underlying data
- [ ] detect configuration of distance
- [ ] for each segment find the optimal route
- [ ] advanced: detect several  routes that are not equivalent
- [ ] create a web interface


### labelling tool installation
устанавливается эта штука не очень удобно - нужно сначала поставить себе анаконду (https://www.anaconda.com/products/individual), а потом либо написать мне, либо проследовать англоязычной инструкции вот тут https://github.com/wkentaro/labelme#windows, а запускается потом по тому как написано тут https://github.com/wkentaro/labelme#usage
не надо этого бояться, я готов провести сквозь этот процесс

### data overview for segmentation
X: map, image.
y: json files with information on map segment
data types:
- line: a single line
- lines: multiple lines connected with each other
- polygon: polygon
to convert to segmentation mask:
* line: every pixel with distance less than 3 to line
* polygon: check if pixel is in polygon
Also add augmentation for data


### get data
- [ ] manual labelling
- [ ] sampling
- [ ] augmentation

### detect runnable map
approaches:
- [ ] cnn
- [ ] u-net
- [ ] rule based approach
- [ ] red lines removal + rules
- [ ] ???
currently have u-net with decent quality

### detect configuration of distance
- [ ] detect triangles
- [ ] detect lines
- [ ] detect circles
- [ ] create energy of configuration and minimize the energy

### optimal route
- [ ] A*
- [ ] bfs

### web interface
- [ ] correct data
- [ ] visualize results


