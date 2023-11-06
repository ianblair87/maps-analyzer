import React from 'react';
import { Stage, Layer, Image, Circle } from 'react-konva';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";


const MainViewer = ({manager}) => {


  return (
    <div style={{borderLeft: "2", borderLeftColor: "black"}}>
      {manager.getSelectedImage() &&
      <TransformWrapper>
      <TransformComponent>
        <Stage
          width={manager.getSelectedImage().width}
          height={manager.getSelectedImage().height}
        >
          <Layer>
            <Image
              image={manager.getSelectedImage()}
            />         
          </Layer>
          <Layer>
            <Circle radius={manager.getCircleRadius()}
              strokeWidth={5}
              x={500}
              y={500}
              stroke={"red"}
            >
            </Circle>
          </Layer>
        </Stage>
      
      </TransformComponent>
      </TransformWrapper>}
      {manager.getSelectedImage() == null && <p> upload image first</p>}
    </div>
  );
};

export default MainViewer;