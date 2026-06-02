// Camera capture loop. We request the highest reasonable resolution the
// sensor will give and capture at its native aspect ratio (capped at
// CAPTURE_LONG_EDGE on the long side). YOLO + the template classifier run on
// a small downscale (done in detector.js); OCR + the symbol classifier crop
// their regions from THIS full-resolution frame so they see real detail —
// a collector_id line is only ~3% of card height, so capture resolution is
// the whole ballgame for reading it.
export const CAPTURE_LONG_EDGE = 1536;

export async function initCamera(onFrame) {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      facingMode: "environment",
    },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();

  // Best-effort: request continuous autofocus so handheld frames sharpen
  // quickly (the OCR only fires on sharp frames via detector.js's gate).
  try {
    const track = stream.getVideoTracks()[0];
    const caps = track.getCapabilities?.() || {};
    if (caps.focusMode && caps.focusMode.includes("continuous")) {
      await track.applyConstraints({ advanced: [{ focusMode: "continuous" }] });
    }
  } catch (_) { /* unsupported — ignore */ }

  // Size the capture canvas to the camera's native aspect, capped on the long
  // edge. Preserving aspect (vs the old 320×320 square) avoids distorting the
  // card before it ever reaches the models.
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const scale = Math.min(1, CAPTURE_LONG_EDGE / Math.max(vw, vh));
  canvas.width = Math.round(vw * scale);
  canvas.height = Math.round(vh * scale);

  function loop() {
    if (!video.paused && !video.ended) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      onFrame(ctx, canvas);
      requestAnimationFrame(loop);
    }
  }

  requestAnimationFrame(loop);
}
