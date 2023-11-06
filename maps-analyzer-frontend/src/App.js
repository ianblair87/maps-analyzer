import MainViewer from "./containers/MainViewer";
import SidePanel from "./containers/SidePanel";
import DrawManager from "./logic/DrawManager";

function App() {
  const manager = DrawManager();
  return (
    <div className="App">
      <h1>Maps analyzer</h1>
      <div style={{display: "flex"}}>
        <SidePanel manager={manager} style={{width: "30%"}}/>
        <MainViewer manager={manager}/> 
      </div>
    </div>
  );
}

export default App;
