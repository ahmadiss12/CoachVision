import { useRouter } from 'expo-router';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Platform,
    Pressable,
    StyleSheet,
    Text,
    View,
} from 'react-native';
import { WebView } from 'react-native-webview';
import * as Speech from 'expo-speech';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { createMockWorkoutStream } from '../services/mock-workout-stream';
import { createLiveSessionSocket } from '../services/ws/live-session';
import { getExerciseMetadata } from '../constants/exercise-metadata';
import { useAppState, useLiveMetrics } from '../state/app-state';
import { colors } from '../theme/colors';

const LANDMARK_SEND_INTERVAL_MS = 60;
const UI_UPDATE_INTERVAL_MS = 80;
const POSE_IN_FLIGHT_TIMEOUT_MS = 500;
// Allowing more than one unacknowledged pose keeps the send rate governed by
// LANDMARK_SEND_INTERVAL_MS instead of by the network round trip. With a
// single slot the effective rate collapses to 1/RTT on a slow link.
const MAX_POSE_IN_FLIGHT = 2;
const VOICE_CUE_COOLDOWN_MS = 1200;
const SAME_VOICE_CUE_COOLDOWN_MS = 3500;
const FORM_TONE_META = {
    idle: {
        label: 'Ready',
        icon: 'scan-outline',
        color: '#38bdf8',
        soft: 'rgba(56, 189, 248, 0.16)',
        border: 'rgba(56, 189, 248, 0.36)',
    },
    good: {
        label: 'Good form',
        icon: 'checkmark-circle-outline',
        color: '#22c55e',
        soft: 'rgba(34, 197, 94, 0.18)',
        border: 'rgba(34, 197, 94, 0.42)',
    },
    warning: {
        label: 'Needs control',
        icon: 'alert-circle-outline',
        color: '#fbbf24',
        soft: 'rgba(251, 191, 36, 0.18)',
        border: 'rgba(251, 191, 36, 0.42)',
    },
    danger: {
        label: 'Fix form',
        icon: 'warning-outline',
        color: '#ef4444',
        soft: 'rgba(239, 68, 68, 0.18)',
        border: 'rgba(239, 68, 68, 0.48)',
    },
};

function clampPercent(value) {
    if (!Number.isFinite(value)) {
        return 0;
    }
    return Math.max(0, Math.min(100, Math.round(value)));
}

function getFormTone(metrics, liveStatus, webStatus) {
    if (webStatus === 'error' || liveStatus === 'error') {
        return 'danger';
    }
    if (!metrics || liveStatus === 'idle' || liveStatus === 'connecting') {
        return 'idle';
    }
    const feedback = String(metrics.feedback || '').toLowerCase();
    const formName = String(metrics.formName || '').toLowerCase();
    const combined = `${feedback} ${formName}`;
    const confidence = Number(metrics.confidence);

    if (
        combined.includes('no person') ||
        combined.includes('no pose') ||
        combined.includes('not detected') ||
        combined.includes('step back') ||
        combined.includes('lost pose') ||
        combined.includes('unsafe') ||
        combined.includes('wrong') ||
        combined.includes('broken') ||
        combined.includes('collapse') ||
        combined.includes('sag') ||
        combined.includes("don't")
    ) {
        return 'danger';
    }
    if (Number.isFinite(confidence) && confidence > 0 && confidence < 0.35) {
        return 'danger';
    }
    if (
        (Number.isFinite(confidence) && confidence > 0 && confidence < 0.6) ||
        (formName && formName !== 'correct') ||
        combined.includes('adjust') ||
        combined.includes('shallow') ||
        combined.includes('deeper') ||
        combined.includes('lean') ||
        combined.includes('chest') ||
        combined.includes('knees') ||
        combined.includes('heels') ||
        combined.includes('control') ||
        combined.includes('straighten') ||
        combined.includes('align')
    ) {
        return 'warning';
    }
    return 'good';
}

const POSE_WEBVIEW_HTML = String.raw`<!doctype html>
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

function feedbackTone(message) {
    if (!message) {
        return colors.textSecondary;
    }
    const lower = message.toLowerCase();
    if (message.includes('!') || lower.includes("don't") || lower.includes('no pose')) {
        return '#f87171';
    }
    if (lower.includes('good') || lower.includes('great')) {
        return '#4ade80';
    }
    return '#fbbf24';
}

function getLiveStatusText(liveStatus, webStatus) {
    if (liveStatus === 'live') {
        return 'Live AI';
    }
    if (liveStatus === 'paused') {
        return 'Paused';
    }
    if (liveStatus === 'connecting') {
        return 'Connecting';
    }
    if (liveStatus === 'demo') {
        return 'Demo';
    }
    if (webStatus === 'ready') {
        return 'Ready';
    }
    if (webStatus === 'camera') {
        return 'Camera on';
    }
    if (webStatus === 'error') {
        return 'Camera issue';
    }
    return 'Loading AI';
}

function MiniMetric({ icon, label, value }) {
    return (
      <View style={styles.miniMetric}>
        <Ionicons name={icon} size={17} color="rgba(255,255,255,0.74)" />
        <View style={styles.miniMetricCopy}>
          <Text style={styles.miniMetricValue} numberOfLines={1}>{value}</Text>
          <Text style={styles.miniMetricLabel}>{label}</Text>
        </View>
      </View>
    );
}

function LiveTopHud({ exerciseName, difficulty, liveStatus, webStatus, toneMeta }) {
    const statusText = getLiveStatusText(liveStatus, webStatus);
    const isLive = liveStatus === 'live';
    const statusColor = webStatus === 'error' || liveStatus === 'error'
        ? '#ef4444'
        : isLive
            ? '#22c55e'
            : '#fbbf24';
    return (
      <View style={styles.topHud} pointerEvents="none">
        <View style={styles.sessionPill}>
          <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
          <View>
            <Text style={styles.sessionStatus}>{statusText}</Text>
            <Text style={styles.sessionName} numberOfLines={1}>
              {(exerciseName || 'workout').toUpperCase()} - {difficulty || 'beginner'}
            </Text>
          </View>
        </View>

        <View style={[styles.formPill, { borderColor: toneMeta.border, backgroundColor: toneMeta.soft }]}>
          <Ionicons name={toneMeta.icon} size={18} color={toneMeta.color} />
          <Text style={[styles.formPillText, { color: toneMeta.color }]}>{toneMeta.label}</Text>
        </View>
      </View>
    );
}

function CameraStatusOverlay({ webStatus }) {
    if (webStatus === 'camera' || webStatus === 'ready') {
        return null;
    }
    const isError = webStatus === 'error';
    return (
      <View pointerEvents="none" style={styles.cameraStatusWrap}>
        <View style={styles.cameraStatusCard}>
          <Ionicons
            name={isError ? 'videocam-off-outline' : 'videocam-outline'}
            size={22}
            color={isError ? '#fca5a5' : '#bae6fd'}
          />
          <Text style={styles.cameraStatusTitle}>
            {isError ? 'Camera unavailable' : 'Opening camera'}
          </Text>
          <Text style={styles.cameraStatusText}>
            {isError ? 'Check camera permission and reload.' : 'Your live preview will appear here.'}
          </Text>
        </View>
      </View>
    );
}

export function WorkoutLiveScreen() {
    const router = useRouter();
    const webViewRef = useRef(null);
    const streamRef = useRef(null);
    const wsRef = useRef(null);
    const poseInFlightRef = useRef(0);
    const isRunningRef = useRef(false);
    const currentSessionIdRef = useRef(null);
    const metricsRef = useRef(null);
    const lastLandmarkSentAtRef = useRef(0);
    const lastUiUpdateAtRef = useRef(0);
    const lastNoPoseNoticeAtRef = useRef(0);
    const lastSpokenAtRef = useRef(0);
    const lastVoiceKeyRef = useRef(null);
    const isVoiceEnabledRef = useRef(true);
    const formToneRef = useRef('idle');
    const latestEndedSummaryRef = useRef(null);
    const pendingEndResolveRef = useRef(null);
    const [isRunning, setIsRunning] = useState(false);
    const [isDemoMode, setIsDemoMode] = useState(false);
    const [liveStatus, setLiveStatus] = useState('idle');
    const [webStatus, setWebStatus] = useState('loading');
    const {
        authTokens,
        currentSession,
        updateMetrics,
        finishWorkout,
        latestError,
        clearError,
        setLatestError,
    } = useAppState();

    const liveMetrics = useLiveMetrics();
    // Before a session exists the store still holds its idle defaults, so keep
    // the screen's own "press play" placeholder for that case.
    const metrics = useMemo(() => (currentSession ? liveMetrics : null) ?? {
        count: 0,
        rawCount: 0,
        measurementType: currentSession?.config?.measurementType,
        measurementLabel: currentSession?.config?.metricLabel,
        state: 'idle',
        angle: 180,
        feedback: 'Press Play when your full body is visible.',
        progress: 0,
        formName: 'Correct',
        formConfidence: null,
        formSource: null,
        confidence: 0,
    }, [liveMetrics, currentSession]);

    const exerciseName = currentSession?.config?.exerciseName ?? 'squat';
    const difficulty = currentSession?.config?.difficulty ?? 'beginner';
    const exerciseMeta = getExerciseMetadata(exerciseName);
    const metricLabel = metrics.measurementLabel
        || currentSession?.config?.metricLabel
        || exerciseMeta.metricLabel;
    const primaryValue = Number.isFinite(Number(metrics.count)) ? Math.round(Number(metrics.count)) : 0;
    const progressPct = clampPercent(Number(metrics.progress ?? 0) * 100);
    const targetValue = Number(currentSession?.config?.targetReps ?? 0);
    const formTone = useMemo(() => getFormTone(metrics, liveStatus, webStatus), [metrics, liveStatus, webStatus]);
    const toneMeta = FORM_TONE_META[formTone] ?? FORM_TONE_META.idle;
    const progressColor = toneMeta.color;
    const shouldForceMock = process.env.EXPO_PUBLIC_USE_MOCK_STREAM === 'true';

    useEffect(() => {
        isRunningRef.current = isRunning;
    }, [isRunning]);

    useEffect(() => {
        currentSessionIdRef.current = currentSession?.sessionId ?? null;
    }, [currentSession?.sessionId]);

    useEffect(() => {
        metricsRef.current = metrics;
    }, [metrics]);

    useEffect(() => {
        formToneRef.current = formTone;
        webViewRef.current?.postMessage(JSON.stringify({ type: 'overlayTone', tone: formTone }));
    }, [formTone]);

    useEffect(() => () => {
        streamRef.current?.stop();
        wsRef.current?.close();
        Speech.stop();
    }, []);

    const stopStreaming = useCallback(() => {
        streamRef.current?.stop();
        streamRef.current = null;
        poseInFlightRef.current = 0;
    }, []);

    const beginMockStreaming = useCallback((reason) => {
        stopStreaming();
        setIsDemoMode(true);
        setLiveStatus('demo');
        setIsRunning(true);
        if (reason) {
            setLatestError(reason);
        }
        streamRef.current = createMockWorkoutStream((next) => updateMetrics(next));
    }, [setLatestError, stopStreaming, updateMetrics]);

    const speakVoiceCue = useCallback((voice) => {
        const text = typeof voice?.text === 'string' ? voice.text.trim() : '';
        if (!text || !isVoiceEnabledRef.current) {
            return;
        }

        const now = Date.now();
        const voiceKey = voice?.label || text;
        if (
            voiceKey === lastVoiceKeyRef.current &&
            now - lastSpokenAtRef.current < SAME_VOICE_CUE_COOLDOWN_MS
        ) {
            return;
        }
        if (now - lastSpokenAtRef.current < VOICE_CUE_COOLDOWN_MS) {
            return;
        }

        lastVoiceKeyRef.current = voiceKey;
        lastSpokenAtRef.current = now;
        Speech.stop();
        Speech.speak(text, {
            language: 'en-US',
            pitch: 1,
            rate: 0.95,
        });
    }, []);

    const handleLiveError = useCallback((message, code) => {
        if (code === 'MODEL_MISSING') {
            setLatestError('Pose model missing on server. Add pose_landmarker.task under backend/coachvision/ai/.');
            setLiveStatus('error');
            setIsRunning(false);
            stopStreaming();
            return;
        }
        if (shouldForceMock) {
            beginMockStreaming(message || 'Using demo metrics (EXPO_PUBLIC_USE_MOCK_STREAM).');
            return;
        }
        setLatestError(message || 'Live tracking connection failed. Check API URL and backend.');
        setLiveStatus('error');
        setIsRunning(false);
        stopStreaming();
    }, [beginMockStreaming, setLatestError, shouldForceMock, stopStreaming]);

    const sendPoseLandmarks = useCallback((payload) => {
        const sessionId = currentSessionIdRef.current;
        const socket = wsRef.current;
        const now = Date.now();
        if (!isRunningRef.current || !socket || !sessionId || !Array.isArray(payload.landmarks)) {
            return;
        }
        if (now - lastLandmarkSentAtRef.current < LANDMARK_SEND_INTERVAL_MS) {
            return;
        }
        if (now - lastLandmarkSentAtRef.current >= POSE_IN_FLIGHT_TIMEOUT_MS) {
            // Nothing came back for a while: assume the outstanding poses were
            // dropped rather than blocking the stream forever.
            poseInFlightRef.current = 0;
        }
        if (poseInFlightRef.current >= MAX_POSE_IN_FLIGHT) {
            return;
        }

        const sent = socket.send({
            type: 'pose',
            sessionId,
            landmarks: payload.landmarks,
            timestampMs: payload.timestampMs || now,
            clientInferenceMs: payload.inferenceMs,
        });
        if (!sent) {
            return;
        }

        poseInFlightRef.current += 1;
        lastLandmarkSentAtRef.current = now;
    }, []);

    const handleWebViewMessage = useCallback((event) => {
        let payload = null;
        try {
            payload = JSON.parse(event.nativeEvent.data);
        } catch {
            return;
        }

        const now = Date.now();
        switch (payload.type) {
            case 'cameraReady':
                setWebStatus('camera');
                break;
            case 'modelReady':
                setWebStatus('ready');
                break;
            case 'pose':
                setWebStatus('ready');
                sendPoseLandmarks(payload);
                break;
            case 'noPose':
                if (isRunningRef.current && now - lastNoPoseNoticeAtRef.current > 700) {
                    lastNoPoseNoticeAtRef.current = now;
                    updateMetrics({
                        ...metricsRef.current,
                        feedback: 'No person detected - step back so your full body is in frame.',
                    });
                }
                break;
            case 'webError':
                setWebStatus('error');
                setLatestError(payload.message || 'Camera AI view failed.');
                break;
            default:
                break;
        }
    }, [sendPoseLandmarks, setLatestError, updateMetrics]);

    const start = async () => {
        clearError();
        latestEndedSummaryRef.current = null;
        setIsDemoMode(false);
        setLiveStatus('connecting');
        stopStreaming();

        if (shouldForceMock) {
            beginMockStreaming('Demo mode enabled (EXPO_PUBLIC_USE_MOCK_STREAM=true).');
            return;
        }
        if (!authTokens?.accessToken || !currentSession?.sessionId) {
            setLatestError('Start a workout from Workout setup first (sign in required).');
            setLiveStatus('error');
            return;
        }

        const liveSocket = createLiveSessionSocket({
            accessToken: authTokens.accessToken,
            onStarted: () => {
                setIsDemoMode(false);
                setLiveStatus('live');
                setIsRunning(true);
            },
            onMetrics: (payload) => {
                const now = Date.now();
                poseInFlightRef.current = Math.max(0, poseInFlightRef.current - 1);
                const nextMetrics = {
                    count: payload.count,
                    rawCount: payload.rawCount ?? payload.count,
                    measurementType: payload.measurementType || currentSession?.config?.measurementType,
                    measurementLabel: payload.measurementLabel || currentSession?.config?.metricLabel,
                    holdDurationSec: payload.holdDurationSec,
                    totalHoldTimeSec: payload.totalHoldTimeSec,
                    bestHoldSec: payload.bestHoldSec,
                    completedHolds: payload.completedHolds,
                    state: payload.state,
                    angle: Number(payload.angle ?? 180),
                    feedback: payload.feedback || 'Keep going.',
                    progress: Number(payload.progress ?? 0),
                    formName: payload.formName || 'Correct',
                    formConfidence: Number.isFinite(Number(payload.formConfidence))
                        ? Number(payload.formConfidence)
                        : null,
                    formProbabilities: payload.formProbabilities || null,
                    formSource: payload.formSource || null,
                    confidence: Number(payload.confidence ?? 0),
                };
                const previousMetrics = metricsRef.current || {};
                const countChanged = Number(nextMetrics.rawCount ?? nextMetrics.count) !== Number(
                    previousMetrics.rawCount ?? previousMetrics.count ?? 0,
                );
                const phaseChanged = String(nextMetrics.state || '') !== String(previousMetrics.state || '');
                if (!countChanged && !phaseChanged && now - lastUiUpdateAtRef.current < UI_UPDATE_INTERVAL_MS) {
                    return;
                }
                lastUiUpdateAtRef.current = now;
                updateMetrics(nextMetrics);
                speakVoiceCue(payload.voice);
            },
            onNoPose: () => {
                poseInFlightRef.current = Math.max(0, poseInFlightRef.current - 1);
                updateMetrics({
                    ...metricsRef.current,
                    feedback: 'No person detected - step back so your full body is in frame.',
                });
            },
            onEnded: (payload) => {
                latestEndedSummaryRef.current = payload.summary || null;
                pendingEndResolveRef.current?.(latestEndedSummaryRef.current);
                pendingEndResolveRef.current = null;
            },
            onError: (message, code) => {
                poseInFlightRef.current = 0;
                handleLiveError(message, code);
            },
        });

        try {
            await liveSocket.connect();
            wsRef.current = liveSocket;
            const sent = liveSocket.send({
                type: 'start',
                sessionId: currentSession.sessionId,
                exerciseName: currentSession.config.exerciseName,
                difficulty: currentSession.config.difficulty || 'beginner',
                targetSets: currentSession.config.targetSets || 1,
                targetReps: currentSession.config.targetReps || 1,
                externalLoadKg: currentSession.config.readinessContext?.externalLoadKg ?? 0,
                bodyWeightKg: currentSession.config.readinessContext?.bodyWeightKg ?? null,
                readinessContext: currentSession.config.readinessContext ?? {},
            });
            if (!sent) {
                throw new Error('Unable to initialize live workout.');
            }
        } catch (error) {
            wsRef.current = null;
            liveSocket.close();
            handleLiveError(error?.message || 'Could not connect to live workout server.');
        }
    };

    const pause = () => {
        stopStreaming();
        Speech.stop();
        setIsRunning(false);
        setLiveStatus((prev) => (prev === 'live' ? 'paused' : prev));
    };

    const leaveWorkout = () => {
        stopStreaming();
        Speech.stop();
        setIsRunning(false);
        setLiveStatus('idle');
        wsRef.current?.close();
        wsRef.current = null;
        if (typeof router.canGoBack === 'function' && router.canGoBack()) {
            router.back();
            return;
        }
        router.replace('/(app)/(tabs)');
    };

    const endSession = async () => {
        stopStreaming();
        Speech.stop();
        setIsRunning(false);
        setLiveStatus('idle');
        let endedSummary = latestEndedSummaryRef.current;
        if (wsRef.current && currentSession?.sessionId) {
            const ackPromise = new Promise((resolve) => {
                const timer = setTimeout(() => {
                    pendingEndResolveRef.current = null;
                    resolve(latestEndedSummaryRef.current);
                }, 1500);
                pendingEndResolveRef.current = (summary) => {
                    clearTimeout(timer);
                    resolve(summary);
                };
            });
            wsRef.current.send({ type: 'end', sessionId: currentSession.sessionId });
            endedSummary = await ackPromise;
        }
        await finishWorkout({ endedSummary });
        wsRef.current?.close();
        wsRef.current = null;
        router.replace('/(app)/session-summary');
    };

    return (
      <View style={styles.root}>
        <WebView
          ref={webViewRef}
          style={styles.webView}
          source={{ html: POSE_WEBVIEW_HTML, baseUrl: 'https://coachvision.local' }}
          originWhitelist={['*']}
          javaScriptEnabled
          domStorageEnabled
          allowsInlineMediaPlayback
          mediaPlaybackRequiresUserAction={false}
          mediaCapturePermissionGrantType={Platform.OS === 'android' ? 'grant' : undefined}
          mixedContentMode="always"
          onMessage={handleWebViewMessage}
          onLoadEnd={() => {
              webViewRef.current?.postMessage(JSON.stringify({
                  type: 'overlayTone',
                  tone: formToneRef.current,
              }));
          }}
          onError={(event) => {
              setWebStatus('error');
              setLatestError(event.nativeEvent.description || 'Camera AI view failed.');
          }}
        />

        <SafeAreaView pointerEvents="box-none" style={styles.overlay}>
          <View pointerEvents="box-none" style={styles.topOverlay}>
            {isDemoMode ? (
              <View style={styles.demoStrip}>
                <Text style={styles.demoStripText}>Demo metrics - not connected to live AI frames</Text>
              </View>
            ) : null}

            <View style={styles.liveTopRow}>
              <Pressable
                accessibilityLabel="Go back"
                accessibilityRole="button"
                hitSlop={10}
                onPress={leaveWorkout}
                style={({ pressed }) => [styles.liveBackButton, pressed && styles.liveBackButtonPressed]}
              >
                <Ionicons name="chevron-back" size={23} color="#fff" />
              </Pressable>
              <View style={styles.liveTopHudWrap}>
                <LiveTopHud
                  exerciseName={exerciseName}
                  difficulty={difficulty}
                  liveStatus={liveStatus}
                  webStatus={webStatus}
                  toneMeta={toneMeta}
                />
              </View>
            </View>
          </View>

          <CameraStatusOverlay webStatus={webStatus} />

          <View pointerEvents="box-none" style={styles.bottomOverlay}>
            <View style={[styles.bottomPanel, { borderColor: toneMeta.border }]}>
              {metrics.feedback ? (
                <View style={[styles.feedbackCard, { backgroundColor: toneMeta.soft, borderColor: toneMeta.border }]}>
                  <Ionicons name={toneMeta.icon} size={18} color={toneMeta.color} />
                  <Text style={[styles.feedbackText, { color: feedbackTone(metrics.feedback) }]} numberOfLines={2}>
                    {metrics.feedback}
                  </Text>
                </View>
              ) : null}

              <View style={styles.mainMetricRow}>
                <View style={styles.repBlock}>
                  <Text style={styles.repLabel}>{metricLabel}</Text>
                  <Text style={styles.repValue}>{primaryValue}</Text>
                  <Text style={styles.repTarget} numberOfLines={1}>
                    {targetValue > 0 ? `Target ${targetValue} ${metricLabel.toLowerCase()}` : 'No target set'}
                  </Text>
                </View>
                <View style={styles.sideMetrics}>
                  <MiniMetric icon="body-outline" label="Phase" value={String(metrics.state || 'idle')} />
                  <MiniMetric
                    icon="analytics-outline"
                    label="Angle"
                    value={`${Number(metrics.angle ?? 0).toFixed(0)} deg`}
                  />
                </View>
              </View>

              <View style={styles.progressHeader}>
                <Text style={styles.progressText}>Motion quality</Text>
                <Text style={[styles.progressText, { color: progressColor }]}>{progressPct}%</Text>
              </View>
              <View style={styles.progressTrack}>
                <View
                  style={[
                      styles.progressFill,
                      { width: `${progressPct}%`, backgroundColor: progressColor },
                  ]}
                />
              </View>

              {latestError ? <Text style={styles.errorText} numberOfLines={2}>{latestError}</Text> : null}

              <View style={styles.controls}>
                <Pressable style={styles.controlButton} onPress={start}>
                  <Ionicons name={isRunning ? 'refresh' : 'play'} size={16} color={colors.textPrimary} />
                  <Text style={styles.controlText}>{isRunning ? 'Restart' : 'Play'}</Text>
                </Pressable>
                <Pressable
                  style={[styles.controlButton, !isRunning && styles.controlButtonDisabled]}
                  onPress={pause}
                  disabled={!isRunning}
                >
                  <Ionicons name="pause" size={16} color={colors.textPrimary} />
                  <Text style={styles.controlText}>Pause</Text>
                </Pressable>
                <Pressable style={[styles.controlButton, styles.controlButtonDanger]} onPress={endSession}>
                  <Ionicons name="stop" size={16} color="#fff" />
                  <Text style={[styles.controlText, styles.controlTextDanger]}>End</Text>
                </Pressable>
              </View>
            </View>
          </View>
        </SafeAreaView>
      </View>
    );
}

const styles = StyleSheet.create({
    root: {
        flex: 1,
        backgroundColor: '#000',
        overflow: 'hidden',
    },
    webView: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: '#000',
    },
    overlay: {
        ...StyleSheet.absoluteFillObject,
        justifyContent: 'space-between',
        paddingHorizontal: 10,
        paddingTop: 6,
        paddingBottom: 6,
    },
    topOverlay: {
        gap: 6,
    },
    bottomOverlay: {
        width: '100%',
    },
    liveTopRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    liveBackButton: {
        width: 44,
        height: 44,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.18)',
        backgroundColor: 'rgba(3, 7, 18, 0.46)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    liveBackButtonPressed: { opacity: 0.78 },
    liveTopHudWrap: { flex: 1, minWidth: 0 },
    topHud: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 10,
    },
    sessionPill: {
        flex: 1,
        minHeight: 44,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        backgroundColor: 'rgba(3, 7, 18, 0.46)',
        borderRadius: 999,
        paddingVertical: 6,
        paddingHorizontal: 10,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.18)',
    },
    statusDot: {
        width: 10,
        height: 10,
        borderRadius: 999,
    },
    sessionStatus: {
        color: '#fff',
        fontSize: 13,
        fontWeight: '800',
    },
    sessionName: {
        color: 'rgba(255,255,255,0.72)',
        fontSize: 10,
        fontWeight: '700',
        marginTop: 1,
    },
    formPill: {
        width: 94,
        minHeight: 44,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 8,
        borderWidth: 1,
        gap: 2,
    },
    formPillText: {
        fontSize: 10,
        lineHeight: 12,
        fontWeight: '900',
        textAlign: 'center',
    },
    bottomPanel: {
        backgroundColor: 'rgba(3, 7, 18, 0.58)',
        borderRadius: 8,
        borderWidth: 1,
        padding: 10,
        gap: 8,
        shadowColor: '#000',
        shadowOpacity: 0.24,
        shadowRadius: 14,
        shadowOffset: { width: 0, height: 8 },
        elevation: 8,
    },
    cameraStatusWrap: {
        position: 'absolute',
        left: 18,
        right: 18,
        top: '35%',
        alignItems: 'center',
        justifyContent: 'center',
    },
    cameraStatusCard: {
        maxWidth: 260,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.16)',
        backgroundColor: 'rgba(3, 7, 18, 0.68)',
        paddingVertical: 12,
        paddingHorizontal: 14,
        alignItems: 'center',
        gap: 5,
    },
    cameraStatusTitle: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '900',
    },
    cameraStatusText: {
        color: 'rgba(255,255,255,0.68)',
        fontSize: 12,
        fontWeight: '700',
        textAlign: 'center',
    },
    mainMetricRow: {
        flexDirection: 'row',
        gap: 10,
        alignItems: 'stretch',
    },
    repBlock: {
        flex: 1.2,
        justifyContent: 'center',
        minWidth: 0,
    },
    repLabel: {
        color: 'rgba(255,255,255,0.68)',
        fontSize: 11,
        fontWeight: '800',
    },
    repValue: {
        color: '#fff',
        fontSize: 50,
        lineHeight: 54,
        fontWeight: '900',
    },
    repTarget: {
        color: 'rgba(255,255,255,0.64)',
        fontSize: 11,
        fontWeight: '800',
    },
    sideMetrics: {
        width: 118,
        gap: 6,
    },
    miniMetric: {
        flex: 1,
        minHeight: 42,
        borderRadius: 8,
        backgroundColor: 'rgba(15, 23, 42, 0.72)',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.1)',
        paddingHorizontal: 8,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 7,
    },
    miniMetricCopy: { flex: 1, minWidth: 0 },
    miniMetricValue: {
        color: '#fff',
        fontSize: 12,
        fontWeight: '900',
        textTransform: 'capitalize',
    },
    miniMetricLabel: {
        color: 'rgba(255,255,255,0.56)',
        fontSize: 9,
        fontWeight: '800',
        marginTop: 1,
    },
    progressHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    progressText: {
        color: 'rgba(255,255,255,0.78)',
        fontSize: 11,
        fontWeight: '800',
    },
    progressTrack: {
        height: 7,
        backgroundColor: 'rgba(148, 163, 184, 0.28)',
        borderRadius: 4,
        overflow: 'hidden',
    },
    progressFill: { height: '100%', borderRadius: 4 },
    feedbackCard: {
        borderWidth: 1,
        borderRadius: 8,
        paddingVertical: 8,
        paddingHorizontal: 10,
        flexDirection: 'row',
        alignItems: 'flex-start',
        gap: 8,
    },
    feedbackText: {
        flex: 1,
        fontSize: 14,
        lineHeight: 19,
        fontWeight: '800',
    },
    demoStrip: {
        backgroundColor: 'rgba(251, 191, 36, 0.92)',
        borderRadius: 8,
        padding: 8,
        marginBottom: 6,
    },
    demoStripText: { color: '#111', fontSize: 12, fontWeight: '700', textAlign: 'center' },
    controls: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 8,
    },
    controlButton: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        minHeight: 44,
        backgroundColor: 'rgba(30, 41, 59, 0.92)',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.22)',
        paddingHorizontal: 10,
        borderRadius: 8,
    },
    controlButtonDisabled: { opacity: 0.55 },
    controlButtonDanger: {
        backgroundColor: 'rgba(220, 38, 38, 0.92)',
        borderColor: 'rgba(255,255,255,0.28)',
    },
    controlText: { color: colors.textPrimary, fontSize: 13, fontWeight: '800' },
    controlTextDanger: { color: '#fff' },
    errorText: { color: '#fca5a5', fontSize: 12, textAlign: 'center' },
});
