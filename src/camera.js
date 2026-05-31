// Camera capture loop. Capture canvas is 640×640 — 4× pixel area of the
// YOLO model input. YOLO itself runs at 320 (its export size), so the C++
// side resizes the 640 input down to 320 before YOLO inference. Template
// classifier, OCR, and symbol classifier all crop from the full 640
// resolution so they see the maximum available detail.
export const CAPTURE_SIZE = 320;

export async function initCamera(onFrame) {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: CAPTURE_SIZE, height: CAPTURE_SIZE, facingMode: "environment" },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();

  canvas.width = CAPTURE_SIZE;
  canvas.height = CAPTURE_SIZE;

  function loop() {
    if (!video.paused && !video.ended) {
      ctx.drawImage(video, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE);
      onFrame(ctx, canvas);
      requestAnimationFrame(loop);
    }
  }

  requestAnimationFrame(loop);
}
