export async function initWasm() {
  const Module = (window.Module = window.Module || {});
  const modelBase = "static/model/cards";

  Module.locateFile = (path) => {
    const dir = modelBase.substring(0, modelBase.lastIndexOf("/") + 1);
    return dir + path;
  };

  const wasmResponse = await fetch(`${modelBase}.wasm`);
  Module.wasmBinary = await wasmResponse.arrayBuffer();

  await loadScript(`${modelBase}.js`);

  await new Promise((resolve) => {
    Module.onRuntimeInitialized = () => {
      _models_init();
      resolve();
    };
  });
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.body.appendChild(script);
  });
}
