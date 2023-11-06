const SidePanel = ({manager}) => {
    return (
        <div>
            <h3>select a map</h3>
            <input type="file" accept="image/*" onChange={(e) => {
                manager.setImageByUrl(URL.createObjectURL(e.target.files[0]));
            }} />
            <h3>adjust control circle radius</h3>
            <input type="range" name="volume" min="1" max="200" onChange={(e) => {
                console.log(e);
                manager.setCircleRadius(e.target.value);
            }} />
            <p> radius = {manager.getCircleRadius()}</p>


        </div>        
    );
};

export default SidePanel;