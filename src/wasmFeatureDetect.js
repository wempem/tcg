// wasmFeatureDetect.js

function validate(bytes) {
  try {
    return WebAssembly.validate(new Uint8Array(bytes));
  } catch {
    return false;
  }
}

async function validateAsync(bytes) {
  try {
    return WebAssembly.validate(new Uint8Array(bytes));
  } catch {
    return false;
  }
}

async function detectBigInt() {
  try {
    const bytes = new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 126, 1, 126, 3, 2, 1, 0, 7,
      5, 1, 1, 98, 0, 0, 10, 6, 1, 4, 0, 32, 0, 11,
    ]);

    const { instance } = await WebAssembly.instantiate(bytes);
    return instance.exports.b(BigInt(0)) === BigInt(0);
  } catch {
    return false;
  }
}

async function detectThreads() {
  try {
    if (typeof SharedArrayBuffer === "undefined") return false;

    // Check SAB actually works (cross-origin isolation required)
    const channel = new MessageChannel();
    channel.port1.postMessage(new SharedArrayBuffer(1));

    return validate([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 4, 1, 3, 1,
      1, 10, 11, 1, 9, 0, 65, 0, 254, 16, 2, 0, 26, 11,
    ]);
  } catch {
    return false;
  }
}

export async function detectWasmFeatures() {
  return {
    bigInt: await detectBigInt(),

    bulkMemory: validate([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 3, 1, 0, 1,
      10, 14, 1, 12, 0, 65, 0, 65, 0, 65, 0, 252, 10, 0, 0, 11,
    ]),

    exceptions: validate([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 9, 1, 7,
      0, 6, 64, 7, 26, 11, 11,
    ]),

    multiValue: validate([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 0, 2, 127, 127, 3, 2, 1, 0, 10,
      8, 1, 6, 0, 65, 0, 65, 0, 11,
    ]),

    mutableGlobals: validate([
      0, 97, 115, 109, 1, 0, 0, 0, 2, 8, 1, 1, 97, 1, 98, 3, 127, 1, 6, 6, 1,
      127, 1, 65, 0, 11, 7, 5, 1, 1, 97, 3, 1,
    ]),

    referenceTypes: validate([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 7, 1, 5,
      0, 208, 112, 26, 11,
    ]),

    saturatedFloatToInt: validate([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 12, 1, 10,
      0, 67, 0, 0, 0, 0, 252, 0, 26, 11,
    ]),

    signExtensions: validate([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 8, 1, 6,
      0, 65, 0, 192, 26, 11,
    ]),

    simd: validate([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 9, 1, 7,
      0, 65, 0, 253, 15, 26, 11,
    ]),

    tailCall: validate([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 6, 1, 4,
      0, 18, 0, 11,
    ]),

    threads: await detectThreads(),
  };
}
