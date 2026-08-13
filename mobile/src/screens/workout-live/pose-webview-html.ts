/**
 * The pose-detection view rendered inside the live workout WebView.
 *
 * MediaPipe runs on-device here rather than on the server: this page owns the
 * camera, detects landmarks, draws the skeleton overlay, and posts landmark
 * frames back to the React Native side. See WS_CONTRACT.md for what the native
 * side then sends over the socket.
 *
 * Extracted verbatim from WorkoutLiveScreen so the screen stays readable.
 */
export const POSE_WEBVIEW_HTML = String.raw`<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta
    name="viewport"
    content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover"
  />
  <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin />
  <link rel="preconnect" href="https://storage.googleapis.com" crossorigin />
  <link rel="dns-prefetch" href="https://cdn.jsdelivr.net" />
  <link rel="dns-prefetch" href="https://storage.googleapis.com" />
  <link rel="modulepreload" href="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35" />
  <style>
    html {
      height: 100%;
      min-height: -webkit-fill-available;
    }
    html, body {
      margin: 0;
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      min-height: 100vh;
      min-height: -webkit-fill-available;
      overflow: hidden;
      background: #000;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #stage {
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      min-height: 100vh;
      min-height: -webkit-fill-available;
      overflow: hidden;
      background: #000;
    }
    #video, #overlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      min-width: 100%;
      min-height: 100%;
      object-fit: cover;
    }
    #video {
      transform: scaleX(-1);
    }
    #overlay {
      pointer-events: none;
    }
    #status {
      position: fixed;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      max-width: 78vw;
      padding: 10px 12px;
      border-radius: 8px;
      color: white;
      background: rgba(0, 0, 0, 0.72);
      font-size: 13px;
      font-weight: 700;
      text-align: center;
      transition: opacity 240ms ease;
    }
  </style>
</head>
<body>
  <div id="stage">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="overlay"></canvas>
    <div id="status">Loading camera and AI...</div>
  </div>

  <script type="module">
    import {
      FilesetResolver,
      PoseLandmarker
    } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35";

    const MODEL_URL =
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
    const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
    const DETECT_INTERVAL_MS = 50;
    const POST_INTERVAL_MS = 60;
    const MIN_VISIBILITY = 0.35;
    const CONNECTIONS = [
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
      [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
      [24, 26], [26, 28], [27, 31], [28, 32]
    ];
    const OVERLAY_PALETTES = {
      idle: {
        stroke: "rgba(56, 189, 248, 0.94)",
        joint: "rgba(186, 230, 253, 0.96)",
        glow: "rgba(56, 189, 248, 0.26)"
      },
      good: {
        stroke: "rgba(34, 197, 94, 0.96)",
        joint: "rgba(187, 247, 208, 0.96)",
        glow: "rgba(34, 197, 94, 0.26)"
      },
      warning: {
        stroke: "rgba(251, 191, 36, 0.98)",
        joint: "rgba(254, 240, 138, 0.98)",
        glow: "rgba(251, 191, 36, 0.3)"
      },
      danger: {
        stroke: "rgba(239, 68, 68, 0.98)",
        joint: "rgba(254, 202, 202, 0.98)",
        glow: "rgba(239, 68, 68, 0.34)"
      }
    };

    const video = document.getElementById("video");
    const canvas = document.getElementById("overlay");
    const stage = document.getElementById("stage");
    const status = document.getElementById("status");
    const ctx = canvas.getContext("2d");
    let canvasCssWidth = 1;
    let canvasCssHeight = 1;
    let landmarker = null;
    let lastDetectAt = 0;
    let lastPostAt = 0;
    let lastVideoTime = -1;
    let overlayTone = "idle";
    const detectTimes = [];

    function post(payload) {
      if (window.ReactNativeWebView) {
        window.ReactNativeWebView.postMessage(JSON.stringify(payload));
      }
    }

    function recentFps(times, now, windowMs) {
      while (times.length && now - times[0] > windowMs) {
        times.shift();
      }
      return Math.round((times.length * 1000) / windowMs);
    }

    function fitCanvas() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const viewport = window.visualViewport || null;
      canvasCssWidth = Math.max(1, Math.round(viewport ? viewport.width : window.innerWidth));
      canvasCssHeight = Math.max(1, Math.round(viewport ? viewport.height : window.innerHeight));
      stage.style.width = canvasCssWidth + "px";
      stage.style.height = canvasCssHeight + "px";
      video.style.width = canvasCssWidth + "px";
      video.style.height = canvasCssHeight + "px";
      canvas.width = Math.round(canvasCssWidth * dpr);
      canvas.height = Math.round(canvasCssHeight * dpr);
      canvas.style.width = canvasCssWidth + "px";
      canvas.style.height = canvasCssHeight + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function visibilityOf(point) {
      return point && Number.isFinite(point.visibility) ? point.visibility : 1;
    }

    function screenPoint(point) {
      return {
        x: (1 - point.x) * canvasCssWidth,
        y: point.y * canvasCssHeight
      };
    }

    function overlayStyle() {
      return OVERLAY_PALETTES[overlayTone] || OVERLAY_PALETTES.good;
    }

    function setOverlayTone(tone) {
      if (OVERLAY_PALETTES[tone]) {
        overlayTone = tone;
      }
    }

    function handleNativeMessage(event) {
      try {
        const payload = JSON.parse(event.data);
        if (payload && payload.type === "overlayTone") {
          setOverlayTone(payload.tone);
        }
      } catch {
        // Ignore non-JSON messages from the native wrapper.
      }
    }

    window.addEventListener("message", handleNativeMessage);
    document.addEventListener("message", handleNativeMessage);

    function expandedPoints(landmarks) {
      const points = landmarks.map(screenPoint);
      const visible = points.filter((point, index) => visibilityOf(landmarks[index]) >= MIN_VISIBILITY);
      if (visible.length < 2) {
        return points;
      }

      const minX = Math.min(...visible.map((point) => point.x));
      const maxX = Math.max(...visible.map((point) => point.x));
      const minY = Math.min(...visible.map((point) => point.y));
      const maxY = Math.max(...visible.map((point) => point.y));
      const centerX = (minX + maxX) / 2;
      const centerY = (minY + maxY) / 2;
      const scale = 1.045;
      return points.map((point) => ({
        x: centerX + (point.x - centerX) * scale,
        y: centerY + (point.y - centerY) * scale
      }));
    }

    function drawPose(landmarks) {
      ctx.clearRect(0, 0, canvasCssWidth, canvasCssHeight);
      if (!landmarks) {
        return;
      }

      const style = overlayStyle();
      const points = expandedPoints(landmarks);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      const visiblePairs = [];
      for (const pair of CONNECTIONS) {
        const a = landmarks[pair[0]];
        const b = landmarks[pair[1]];
        if (!a || !b || visibilityOf(a) < MIN_VISIBILITY || visibilityOf(b) < MIN_VISIBILITY) {
          continue;
        }
        visiblePairs.push(pair);
      }

      ctx.shadowBlur = 0;
      ctx.lineWidth = 10;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.34)";
      ctx.beginPath();
      for (const pair of visiblePairs) {
        const pa = points[pair[0]];
        const pb = points[pair[1]];
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
      }
      ctx.stroke();

      ctx.shadowBlur = 6;
      ctx.shadowColor = style.glow;
      ctx.lineWidth = 6;
      ctx.strokeStyle = style.stroke;
      ctx.beginPath();
      for (const pair of visiblePairs) {
        const pa = points[pair[0]];
        const pb = points[pair[1]];
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
      }
      ctx.stroke();

      ctx.shadowBlur = 4;
      ctx.shadowColor = style.glow;
      ctx.fillStyle = style.joint;
      for (let index = 0; index < landmarks.length; index += 1) {
        const landmark = landmarks[index];
        if (visibilityOf(landmark) < MIN_VISIBILITY) {
          continue;
        }
        const p = points[index];
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6.25, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    }

    async function startCamera() {
      status.textContent = "Opening camera...";
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API is not available in this WebView.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 }
        }
      });
      video.srcObject = stream;
      await video.play();
      post({
        type: "cameraReady",
        videoWidth: video.videoWidth || 0,
        videoHeight: video.videoHeight || 0
      });
    }

    async function startModel() {
      status.textContent = "Loading on-device AI...";
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      post({ type: "modelReady" });
    }

    function runPoseLoop(now) {
      requestAnimationFrame(runPoseLoop);
      if (!landmarker || video.readyState < 2) {
        return;
      }
      if (now - lastDetectAt < DETECT_INTERVAL_MS || video.currentTime === lastVideoTime) {
        return;
      }

      lastVideoTime = video.currentTime;
      lastDetectAt = now;
      const startedAt = performance.now();
      let result = null;
      try {
        result = landmarker.detectForVideo(video, now);
      } catch (error) {
        post({
          type: "webError",
          message: error && error.message ? error.message : String(error)
        });
        return;
      }

      const inferenceMs = Math.round(performance.now() - startedAt);
      detectTimes.push(now);
      const localAiFps = recentFps(detectTimes, now, 3000);
      const landmarks = result && result.landmarks && result.landmarks[0]
        ? result.landmarks[0]
        : null;
      drawPose(landmarks);

      if (now - lastPostAt < POST_INTERVAL_MS) {
        return;
      }
      lastPostAt = now;

      if (!landmarks) {
        post({
          type: "noPose",
          timestampMs: Date.now(),
          localAiFps,
          inferenceMs
        });
        return;
      }

      post({
        type: "pose",
        timestampMs: Date.now(),
        localAiFps,
        inferenceMs,
        landmarks: landmarks.map((landmark) => [
          landmark.x,
          landmark.y,
          visibilityOf(landmark)
        ])
      });
    }

    async function boot() {
      try {
        fitCanvas();
        status.textContent = "Opening camera and loading AI...";
        await Promise.all([startCamera(), startModel()]);
        status.textContent = "On-device AI ready";
        setTimeout(() => {
          status.style.opacity = "0";
        }, 650);
        requestAnimationFrame(runPoseLoop);
      } catch (error) {
        const message = error && error.message ? error.message : String(error);
        status.textContent = message;
        post({ type: "webError", message });
      }
    }

    window.addEventListener("resize", fitCanvas);
    boot();
  </script>
</body>
</html>`;
