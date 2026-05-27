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
import { useAppState } from '../state/app-state';
import { colors } from '../theme/colors';

const LANDMARK_SEND_INTERVAL_MS = 80;
const UI_UPDATE_INTERVAL_MS = 120;
const POSE_IN_FLIGHT_TIMEOUT_MS = 500;
const VOICE_CUE_COOLDOWN_MS = 1200;
const SAME_VOICE_CUE_COOLDOWN_MS = 3500;

const POSE_WEBVIEW_HTML = String.raw`<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta
    name="viewport"
    content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover"
  />
  <style>
    html, body {
      margin: 0;
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      background: #000;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #stage {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      background: #000;
    }
    #video, #overlay {
      position: absolute;
      inset: 0;
      width: 100vw;
      height: 100vh;
      height: 100dvh;
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
    const DETECT_INTERVAL_MS = 66;
    const POST_INTERVAL_MS = 80;
    const MIN_VISIBILITY = 0.35;
    const CONNECTIONS = [
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
      [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
      [24, 26], [26, 28], [27, 31], [28, 32]
    ];

    const video = document.getElementById("video");
    const canvas = document.getElementById("overlay");
    const status = document.getElementById("status");
    const ctx = canvas.getContext("2d");
    let canvasCssWidth = 1;
    let canvasCssHeight = 1;
    let landmarker = null;
    let lastDetectAt = 0;
    let lastPostAt = 0;
    let lastVideoTime = -1;
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
      canvasCssWidth = Math.max(1, window.innerWidth);
      canvasCssHeight = Math.max(1, window.innerHeight);
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

    function drawPose(landmarks) {
      ctx.clearRect(0, 0, canvasCssWidth, canvasCssHeight);
      if (!landmarks) {
        return;
      }

      ctx.lineWidth = 4;
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(74, 222, 128, 0.92)";
      for (const pair of CONNECTIONS) {
        const a = landmarks[pair[0]];
        const b = landmarks[pair[1]];
        if (!a || !b || visibilityOf(a) < MIN_VISIBILITY || visibilityOf(b) < MIN_VISIBILITY) {
          continue;
        }
        const pa = screenPoint(a);
        const pb = screenPoint(b);
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }

      ctx.fillStyle = "rgba(251, 191, 36, 0.95)";
      for (const landmark of landmarks) {
        if (visibilityOf(landmark) < MIN_VISIBILITY) {
          continue;
        }
        const p = screenPoint(landmark);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    async function startCamera() {
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
      status.textContent = "On-device AI ready";
      setTimeout(() => {
        status.style.opacity = "0";
      }, 900);
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
        await startCamera();
        await startModel();
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

function StatChip({ label, value }) {
    return (
      <View style={styles.statChip}>
        <Text style={styles.statValue}>{value}</Text>
        <Text style={styles.statLabel}>{label}</Text>
      </View>
    );
}

function LiveTopHud({ exerciseName, difficulty, liveStatus, webStatus, perfStats }) {
    const statusText = getLiveStatusText(liveStatus, webStatus);
    const isLive = liveStatus === 'live';
    return (
      <View style={styles.topHud} pointerEvents="none">
        <View style={styles.sessionPill}>
          <View style={[styles.statusDot, isLive ? styles.statusDotLive : styles.statusDotIdle]} />
          <View>
            <Text style={styles.sessionStatus}>{statusText}</Text>
            <Text style={styles.sessionName}>
              {(exerciseName || 'workout').toUpperCase()} - {difficulty || 'beginner'}
            </Text>
          </View>
        </View>

        <View style={styles.fpsPill}>
          <Text style={styles.fpsValue}>{perfStats.aiFps || 0}</Text>
          <Text style={styles.fpsLabel}>AI FPS</Text>
        </View>
      </View>
    );
}

function VoiceToggle({ enabled, onToggle }) {
    return (
      <Pressable style={styles.voiceToggle} onPress={onToggle}>
        <Ionicons
          name={enabled ? 'volume-high' : 'volume-mute'}
          size={16}
          color={enabled ? '#4ade80' : 'rgba(255,255,255,0.62)'}
        />
        <Text style={[styles.voiceToggleText, enabled ? styles.voiceToggleTextOn : null]}>
          {enabled ? 'Voice on' : 'Voice off'}
        </Text>
      </Pressable>
    );
}

export function WorkoutLiveScreen() {
    const router = useRouter();
    const streamRef = useRef(null);
    const wsRef = useRef(null);
    const isFrameInFlightRef = useRef(false);
    const isRunningRef = useRef(false);
    const currentSessionIdRef = useRef(null);
    const metricsRef = useRef(null);
    const lastLandmarkSentAtRef = useRef(0);
    const lastUiUpdateAtRef = useRef(0);
    const lastNoPoseNoticeAtRef = useRef(0);
    const lastSpokenAtRef = useRef(0);
    const lastVoiceKeyRef = useRef(null);
    const isVoiceEnabledRef = useRef(true);
    const metricsTimesRef = useRef([]);
    const uiTimesRef = useRef([]);
    const latestEndedSummaryRef = useRef(null);
    const pendingEndResolveRef = useRef(null);
    const [isRunning, setIsRunning] = useState(false);
    const [isDemoMode, setIsDemoMode] = useState(false);
    const [liveStatus, setLiveStatus] = useState('idle');
    const [webStatus, setWebStatus] = useState('loading');
    const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
    const [perfStats, setPerfStats] = useState({
        aiFps: 0,
        serverFps: 0,
        uiFps: 0,
        latencyMs: 0,
    });
    const {
        authTokens,
        currentSession,
        updateMetrics,
        finishWorkout,
        latestError,
        clearError,
        setLatestError,
    } = useAppState();

    const metrics = useMemo(() => currentSession?.latestMetrics ?? {
        count: 0,
        rawCount: 0,
        measurementType: currentSession?.config?.measurementType,
        measurementLabel: currentSession?.config?.metricLabel,
        state: 'idle',
        angle: 180,
        feedback: 'Press Play when your full body is visible.',
        progress: 0,
        formName: 'Correct',
        confidence: 0,
    }, [
        currentSession?.latestMetrics,
        currentSession?.config?.measurementType,
        currentSession?.config?.metricLabel,
    ]);

    const exerciseName = currentSession?.config?.exerciseName ?? 'squat';
    const difficulty = currentSession?.config?.difficulty ?? 'beginner';
    const exerciseMeta = getExerciseMetadata(exerciseName);
    const metricLabel = metrics.measurementLabel
        || currentSession?.config?.metricLabel
        || exerciseMeta.metricLabel;
    const primaryValue = Math.round(Number(metrics.count ?? 0));
    const confidencePct = Math.round((metrics.confidence ?? 0) * 100);
    const progressPct = Math.round(metrics.progress * 100);
    const progressColor = metrics.progress < 0.5 ? '#fbbf24' : '#4ade80';
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
        isVoiceEnabledRef.current = isVoiceEnabled;
        if (!isVoiceEnabled) {
            Speech.stop();
        }
    }, [isVoiceEnabled]);

    useEffect(() => () => {
        streamRef.current?.stop();
        wsRef.current?.close();
        Speech.stop();
    }, []);

    const countRecent = (times, now, windowMs = 3000) => {
        while (times.length && now - times[0] > windowMs) {
            times.shift();
        }
        return Math.round((times.length * 1000) / windowMs);
    };

    const stopStreaming = useCallback(() => {
        streamRef.current?.stop();
        streamRef.current = null;
        isFrameInFlightRef.current = false;
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
        if (
            isFrameInFlightRef.current &&
            now - lastLandmarkSentAtRef.current < POSE_IN_FLIGHT_TIMEOUT_MS
        ) {
            return;
        }

        const sent = socket.send({
            type: 'pose',
            sessionId,
            landmarks: payload.landmarks,
            timestampMs: payload.timestampMs || now,
            clientInferenceMs: payload.inferenceMs,
            localAiFps: payload.localAiFps,
        });
        if (!sent) {
            return;
        }

        isFrameInFlightRef.current = true;
        lastLandmarkSentAtRef.current = now;
        setPerfStats((prev) => ({
            ...prev,
            aiFps: Number(payload.localAiFps || prev.aiFps || 0),
        }));
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
                setPerfStats((prev) => ({
                    ...prev,
                    aiFps: Number(payload.localAiFps || prev.aiFps || 0),
                }));
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
                isFrameInFlightRef.current = false;
                metricsTimesRef.current.push(now);
                const latencyMs = lastLandmarkSentAtRef.current
                    ? Math.max(0, now - lastLandmarkSentAtRef.current)
                    : 0;
                const serverFps = countRecent(metricsTimesRef.current, now);
                if (now - lastUiUpdateAtRef.current < UI_UPDATE_INTERVAL_MS) {
                    setPerfStats((prev) => ({
                        ...prev,
                        serverFps,
                        latencyMs,
                    }));
                    return;
                }
                lastUiUpdateAtRef.current = now;
                uiTimesRef.current.push(now);
                setPerfStats((prev) => ({
                    ...prev,
                    serverFps,
                    uiFps: countRecent(uiTimesRef.current, now),
                    latencyMs,
                }));
                updateMetrics({
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
                    confidence: Number(payload.confidence ?? 0),
                });
                speakVoiceCue(payload.voice);
            },
            onNoPose: () => {
                isFrameInFlightRef.current = false;
                const now = Date.now();
                const latencyMs = lastLandmarkSentAtRef.current
                    ? Math.max(0, now - lastLandmarkSentAtRef.current)
                    : 0;
                metricsTimesRef.current.push(now);
                setPerfStats((prev) => ({
                    ...prev,
                    serverFps: countRecent(metricsTimesRef.current, now),
                    latencyMs,
                }));
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
                isFrameInFlightRef.current = false;
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
          onError={(event) => {
              setWebStatus('error');
              setLatestError(event.nativeEvent.description || 'Camera AI view failed.');
          }}
        />

        <SafeAreaView pointerEvents="box-none" style={styles.overlay}>
          {isDemoMode ? (
            <View style={styles.demoStrip}>
              <Text style={styles.demoStripText}>Demo metrics - not connected to live AI frames</Text>
            </View>
          ) : null}

          <LiveTopHud
            exerciseName={exerciseName}
            difficulty={difficulty}
            liveStatus={liveStatus}
            webStatus={webStatus}
            perfStats={perfStats}
          />

          <View style={styles.bottomPanel}>
              <View style={styles.repRow}>
              <View style={styles.repBlock}>
                <Text style={styles.repLabel}>{metricLabel}</Text>
                <Text style={styles.repValue}>{primaryValue}</Text>
              </View>
              <View style={styles.motionBlock}>
                <Text style={styles.motionState}>{metrics.state}</Text>
                <Text style={styles.motionAngle}>{metrics.angle.toFixed(1)} deg</Text>
              </View>
            </View>

            <View style={styles.progressHeader}>
              <Text style={styles.progressText}>{metrics.formName}</Text>
              <Text style={styles.progressText}>{progressPct}%</Text>
            </View>
            <View style={styles.progressTrack}>
              <View
                style={[
                    styles.progressFill,
                    { width: `${progressPct}%`, backgroundColor: progressColor },
                ]}
              />
            </View>

            {metrics.feedback ? (
              <Text style={[styles.feedbackText, { color: feedbackTone(metrics.feedback) }]}>
                {metrics.feedback}
              </Text>
            ) : null}

            {latestError ? <Text style={styles.errorText}>{latestError}</Text> : null}

            <View style={styles.statsRow}>
              <StatChip label="Detection" value={`${confidencePct}%`} />
              <StatChip label="Server" value={perfStats.serverFps} />
              <StatChip label="UI" value={perfStats.uiFps} />
              <StatChip label="Latency" value={`${perfStats.latencyMs}ms`} />
            </View>

            <VoiceToggle
              enabled={isVoiceEnabled}
              onToggle={() => setIsVoiceEnabled((prev) => !prev)}
            />

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
        flex: 1,
        justifyContent: 'space-between',
        paddingHorizontal: 12,
        paddingTop: 8,
        paddingBottom: 10,
    },
    topHud: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 10,
    },
    sessionPill: {
        flex: 1,
        minHeight: 54,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
        backgroundColor: 'rgba(3, 7, 18, 0.66)',
        borderRadius: 999,
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.18)',
    },
    statusDot: {
        width: 10,
        height: 10,
        borderRadius: 999,
    },
    statusDotLive: { backgroundColor: '#22c55e' },
    statusDotIdle: { backgroundColor: '#f59e0b' },
    sessionStatus: {
        color: '#fff',
        fontSize: 15,
        fontWeight: '800',
    },
    sessionName: {
        color: 'rgba(255,255,255,0.72)',
        fontSize: 11,
        fontWeight: '700',
        marginTop: 1,
    },
    fpsPill: {
        width: 74,
        minHeight: 54,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(15, 23, 42, 0.72)',
        borderRadius: 8,
        borderWidth: 1,
        borderColor: 'rgba(74, 222, 128, 0.38)',
    },
    fpsValue: {
        color: '#4ade80',
        fontSize: 20,
        fontWeight: '900',
    },
    fpsLabel: {
        color: 'rgba(255,255,255,0.7)',
        fontSize: 10,
        fontWeight: '800',
    },
    bottomPanel: {
        backgroundColor: 'rgba(3, 7, 18, 0.78)',
        borderRadius: 8,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.16)',
        padding: 12,
        gap: 10,
    },
    repRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
    },
    repBlock: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        gap: 9,
    },
    repLabel: {
        color: 'rgba(255,255,255,0.68)',
        fontSize: 12,
        fontWeight: '800',
        marginBottom: 8,
    },
    repValue: {
        color: '#fff',
        fontSize: 54,
        lineHeight: 58,
        fontWeight: '900',
    },
    motionBlock: {
        alignItems: 'flex-end',
        justifyContent: 'center',
        minWidth: 104,
    },
    motionState: {
        color: '#fbbf24',
        fontSize: 16,
        fontWeight: '900',
    },
    motionAngle: {
        color: 'rgba(255,255,255,0.7)',
        fontSize: 12,
        fontWeight: '700',
        marginTop: 2,
    },
    progressHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    progressText: {
        color: 'rgba(255,255,255,0.78)',
        fontSize: 12,
        fontWeight: '800',
    },
    progressTrack: {
        height: 8,
        backgroundColor: 'rgba(148, 163, 184, 0.28)',
        borderRadius: 4,
        overflow: 'hidden',
    },
    progressFill: { height: '100%', borderRadius: 4 },
    feedbackText: {
        fontSize: 16,
        lineHeight: 21,
        fontWeight: '800',
        textAlign: 'center',
    },
    statsRow: {
        flexDirection: 'row',
        gap: 6,
    },
    statChip: {
        flex: 1,
        minHeight: 46,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(15, 23, 42, 0.7)',
        borderRadius: 8,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.1)',
    },
    statValue: {
        color: '#fff',
        fontSize: 15,
        fontWeight: '900',
    },
    statLabel: {
        color: 'rgba(255,255,255,0.58)',
        fontSize: 10,
        fontWeight: '800',
        marginTop: 1,
    },
    voiceToggle: {
        minHeight: 38,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 7,
        backgroundColor: 'rgba(15, 23, 42, 0.62)',
        borderRadius: 8,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.12)',
    },
    voiceToggleText: {
        color: 'rgba(255,255,255,0.62)',
        fontSize: 12,
        fontWeight: '800',
    },
    voiceToggleTextOn: {
        color: '#4ade80',
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
