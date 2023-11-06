import { useState, useEffect } from "react";

const DrawManager = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [circleRadius, setCircleRadius] = useState(1);

    const getSelectedImage = () => {
        return selectedImage;
    };
    const getCircleRadius = () => circleRadius;

    const setImageByUrl = (url) => {
        const img = new window.Image();
        img.src = url; 
        img.onload = () => {
            setSelectedImage(img); 
        };
    }

    useEffect(
        () => {
            setImageByUrl("http://o-mephi.net/cup/prot/polunocnoe_2023_cont1.jpg");
        },
        []
    );

    return {
        getSelectedImage,
        setImageByUrl,
        getCircleRadius,
        setCircleRadius
    };
};

export default DrawManager;
