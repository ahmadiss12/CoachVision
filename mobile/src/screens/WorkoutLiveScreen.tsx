import { Ionicons } from '@expo/vector-icons';
import * as Speech from 'expo-speech';
import { useRouter } from 'expo-router';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Platform,
    Pressable,
    StyleSheet,
    Text,
    View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView, type WebViewMessageEvent } from 'react-native-webview';

import { getExerciseMetadata } from '../constants/exercise-metadata';
import { createMockWorkoutStream } from '../services/mock-workout-stream';
import { createLiveSessionSocket, type LiveSessionSocket } from '../services/ws/live-session';
import type { ErrorMessage, MetricsMessage, VoiceCue } from '../services/ws/messages';
import { useAppState, useLiveMetrics } from '../state/app-state';
import type { AppStateSlice, CurrentSession } from '../state/app-state-types';
import { metricsFromWire, shouldRenderFrame, type LiveMetrics } from '../state/live-metrics';
import { colors } from '../theme/colors';
import { POSE_WEBVIEW_HTML } from './workout-live/pose-webview-html';
import {
    clampPercent,
    feedbackTone,
    FORM_TONE_META,
    getFormTone,
    getLiveStatusText,
    getStatusColor,
    type FormTone,
    type LiveStatus,
    type ToneMeta,
    type WebStatus,
} from './workout-live/presentation';
import {
    parseWebViewMessage,
    type PoseFrameMessage,
    type WebViewCommand,
} from './workout-live/webview-bridge';

const LANDMARK_SEND_INTERVAL_MS = 60;
const UI_UPDATE_INTERVAL_MS = 80;
const POSE_IN_FLIGHT_TIMEOUT_MS = 500;
// Allowing more than one unacknowledged pose keeps the send rate governed by
// LANDMARK_SEND_INTERVAL_MS instead of by the network round trip. With a
// single slot the effective rate collapses to 1/RTT on a slow link.
const MAX_POSE_IN_FLIGHT = 2;
const VOICE_CUE_COOLDOWN_MS = 1200;
const SAME_VOICE_CUE_COOLDOWN_MS = 3500;
const NO_POSE_NOTICE_INTERVAL_MS = 700;
const END_ACK_TIMEOUT_MS = 1500;

const NO_POSE_FEEDBACK = 'No person detected - step back so your full body is in frame.';

type MiniMetricProps = {
    icon: React.ComponentProps<typeof Ionicons>['name'];
    label: string;
    value: string;
};

function MiniMetric({ icon, label, value }: MiniMetricProps) {
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

type LiveTopHudProps = {
    exerciseName: string;
    difficulty: string;
    liveStatus: LiveStatus;
    webStatus: WebStatus;
    toneMeta: ToneMeta;
};

function LiveTopHud({
    exerciseName,
    difficulty,
    liveStatus,
    webStatus,
    toneMeta,
}: LiveTopHudProps) {
    const statusText = getLiveStatusText(liveStatus, webStatus);
    const statusColor = getStatusColor(liveStatus, webStatus);
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
          <Ionicons
            name={toneMeta.icon as React.ComponentProps<typeof Ionicons>['name']}
            size={18}
            color={toneMeta.color}
          />
          <Text style={[styles.formPillText, { color: toneMeta.color }]}>{toneMeta.label}</Text>
        </View>
      </View>
    );
}

function CameraStatusOverlay({ webStatus }: { webStatus: WebStatus }) {
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
    const webViewRef = useRef<WebView>(null);
    const streamRef = useRef<{ stop: () => void } | null>(null);
    const wsRef = useRef<LiveSessionSocket | null>(null);
    const poseInFlightRef = useRef(0);
    const isRunningRef = useRef(false);
    const currentSessionIdRef = useRef<string | null>(null);
    const metricsRef = useRef<LiveMetrics | null>(null);
    const lastLandmarkSentAtRef = useRef(0);
    const lastUiUpdateAtRef = useRef(0);
    const lastNoPoseNoticeAtRef = useRef(0);
    const lastSpokenAtRef = useRef(0);
    const lastVoiceKeyRef = useRef<string | null>(null);
    const isVoiceEnabledRef = useRef(true);
    const formToneRef = useRef<FormTone>('idle');
    const latestEndedSummaryRef = useRef<Record<string, unknown> | null>(null);
    const pendingEndResolveRef = useRef<((summary: Record<string, unknown> | null) => void) | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [isDemoMode, setIsDemoMode] = useState(false);
    const [liveStatus, setLiveStatus] = useState<LiveStatus>('idle');
    const [webStatus, setWebStatus] = useState<WebStatus>('loading');
    // app-state.jsx is JavaScript and creates its context with
    // createContext(null), so these hooks infer as `never`. The assertion names
    // the shape described in app-state-types.ts; it becomes a real check when
    // that module converts.
    const {
        authTokens,
        currentSession,
        updateMetrics,
        finishWorkout,
        latestError,
        clearError,
        setLatestError,
    } = useAppState() as AppStateSlice;

    const session: CurrentSession | null = currentSession ?? null;
    const liveMetrics = useLiveMetrics() as LiveMetrics | null;

    const postToWebView = useCallback((command: WebViewCommand) => {
        webViewRef.current?.postMessage(JSON.stringify(command));
    }, []);

    // Before a session exists the store still holds its idle defaults, so keep
    // the screen's own "press play" placeholder for that case.
    const metrics = useMemo<LiveMetrics>(() => (session ? liveMetrics : null) ?? {
        count: 0,
        rawCount: 0,
        measurementType: session?.config?.measurementType,
        measurementLabel: session?.config?.metricLabel,
        state: 'idle',
        angle: 180,
        feedback: 'Press Play when your full body is visible.',
        progress: 0,
        formName: 'Correct',
        formConfidence: null,
        formSource: null,
        confidence: 0,
    }, [liveMetrics, session]);

    const exerciseName = session?.config?.exerciseName ?? 'squat';
    const difficulty = session?.config?.difficulty ?? 'beginner';
    const exerciseMeta = getExerciseMetadata(exerciseName);
    const metricLabel = metrics.measurementLabel
        || session?.config?.metricLabel
        || exerciseMeta.metricLabel;
    const primaryValue = Number.isFinite(Number(metrics.count)) ? Math.round(Number(metrics.count)) : 0;
    const progressPct = clampPercent(Number(metrics.progress ?? 0) * 100);
    const targetValue = Number(session?.config?.targetReps ?? 0);
    const formTone = useMemo(
        () => getFormTone(metrics, liveStatus, webStatus),
        [metrics, liveStatus, webStatus],
    );
    const toneMeta = FORM_TONE_META[formTone] ?? FORM_TONE_META.idle;
    const progressColor = toneMeta.color;
    const shouldForceMock = process.env.EXPO_PUBLIC_USE_MOCK_STREAM === 'true';

    useEffect(() => {
        isRunningRef.current = isRunning;
    }, [isRunning]);

    useEffect(() => {
        currentSessionIdRef.current = session?.sessionId ?? null;
    }, [session?.sessionId]);

    useEffect(() => {
        metricsRef.current = metrics;
    }, [metrics]);

    useEffect(() => {
        formToneRef.current = formTone;
        postToWebView({ type: 'overlayTone', tone: formTone });
    }, [formTone, postToWebView]);

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

    const beginMockStreaming = useCallback((reason?: string) => {
        stopStreaming();
        setIsDemoMode(true);
        setLiveStatus('demo');
        setIsRunning(true);
        if (reason) {
            setLatestError(reason);
        }
        streamRef.current = createMockWorkoutStream((next: LiveMetrics) => updateMetrics(next));
    }, [setLatestError, stopStreaming, updateMetrics]);

    const speakVoiceCue = useCallback((voice: VoiceCue | null | undefined) => {
        // voice is {label, text}: label de-duplicates, text is what is spoken.
        // Passing the object itself anywhere near a <Text> is what crashed this
        // screen before the payload was typed.
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

    const handleLiveError = useCallback((message?: string, code?: ErrorMessage['code']) => {
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

    const sendPoseLandmarks = useCallback((
        landmarks: PoseFrameMessage['landmarks'],
        timestampMs: number | undefined,
        inferenceMs: number | undefined,
    ) => {
        const sessionId = currentSessionIdRef.current;
        const socket = wsRef.current;
        const now = Date.now();
        if (!isRunningRef.current || !socket || !sessionId) {
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
            landmarks,
            timestampMs: timestampMs || now,
            ...(inferenceMs === undefined ? {} : { clientInferenceMs: inferenceMs }),
        });
        if (!sent) {
            return;
        }

        poseInFlightRef.current += 1;
        lastLandmarkSentAtRef.current = now;
    }, []);

    const showNoPoseNotice = useCallback(() => {
        const previous = metricsRef.current;
        if (!previous) {
            return;
        }
        updateMetrics({ ...previous, feedback: NO_POSE_FEEDBACK });
    }, [updateMetrics]);

    const handleWebViewMessage = useCallback((event: WebViewMessageEvent) => {
        const payload = parseWebViewMessage(event.nativeEvent.data);
        if (!payload) {
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
                sendPoseLandmarks(payload.landmarks, payload.timestampMs, payload.inferenceMs);
                break;
            case 'noPose':
                if (
                    isRunningRef.current &&
                    now - lastNoPoseNoticeAtRef.current > NO_POSE_NOTICE_INTERVAL_MS
                ) {
                    lastNoPoseNoticeAtRef.current = now;
                    showNoPoseNotice();
                }
                break;
            case 'webError':
                setWebStatus('error');
                setLatestError(payload.message || 'Camera AI view failed.');
                break;
        }
    }, [sendPoseLandmarks, setLatestError, showNoPoseNotice]);

    const handleMetrics = useCallback((payload: MetricsMessage) => {
        const now = Date.now();
        poseInFlightRef.current = Math.max(0, poseInFlightRef.current - 1);

        const nextMetrics = metricsFromWire(payload, {
            measurementType: session?.config?.measurementType,
            measurementLabel: session?.config?.metricLabel,
        });

        if (!shouldRenderFrame(
            nextMetrics,
            metricsRef.current,
            now - lastUiUpdateAtRef.current,
            UI_UPDATE_INTERVAL_MS,
        )) {
            return;
        }

        lastUiUpdateAtRef.current = now;
        updateMetrics(nextMetrics);
        speakVoiceCue(payload.voice);
    }, [session?.config?.measurementType, session?.config?.metricLabel, speakVoiceCue, updateMetrics]);

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
        if (!authTokens?.accessToken || !session?.sessionId || !session.config) {
            setLatestError('Start a workout from Workout setup first (sign in required).');
            setLiveStatus('error');
            return;
        }

        const sessionId = session.sessionId;
        const config = session.config;

        const liveSocket = createLiveSessionSocket({
            accessToken: authTokens.accessToken,
            onStarted: () => {
                setIsDemoMode(false);
                setLiveStatus('live');
                setIsRunning(true);
            },
            onMetrics: handleMetrics,
            onNoPose: () => {
                poseInFlightRef.current = Math.max(0, poseInFlightRef.current - 1);
                showNoPoseNotice();
            },
            onEnded: (payload) => {
                latestEndedSummaryRef.current = payload.summary ?? null;
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
                sessionId,
                exerciseName: config.exerciseName ?? 'squat',
                difficulty: (config.difficulty || 'beginner') as 'beginner' | 'intermediate' | 'advanced',
                targetSets: config.targetSets || 1,
                targetReps: config.targetReps || 1,
                externalLoadKg: config.readinessContext?.externalLoadKg ?? 0,
                bodyWeightKg: config.readinessContext?.bodyWeightKg ?? null,
                readinessContext: config.readinessContext ?? {},
            } as Parameters<LiveSessionSocket['send']>[0]);
            if (!sent) {
                throw new Error('Unable to initialize live workout.');
            }
        } catch (error) {
            wsRef.current = null;
            liveSocket.close();
            handleLiveError(
                error instanceof Error ? error.message : 'Could not connect to live workout server.',
            );
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
        if (wsRef.current && session?.sessionId) {
            const ackPromise = new Promise<Record<string, unknown> | null>((resolve) => {
                const timer = setTimeout(() => {
                    pendingEndResolveRef.current = null;
                    resolve(latestEndedSummaryRef.current);
                }, END_ACK_TIMEOUT_MS);
                pendingEndResolveRef.current = (summary) => {
                    clearTimeout(timer);
                    resolve(summary);
                };
            });
            wsRef.current.send({ type: 'end', sessionId: session.sessionId });
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
          // Spread rather than passing `undefined` on iOS: setting an optional
          // prop to undefined is not the same as omitting it.
          {...(Platform.OS === 'android'
              ? { mediaCapturePermissionGrantType: 'grant' as const }
              : {})}
          mixedContentMode="always"
          onMessage={handleWebViewMessage}
          onLoadEnd={() => {
              postToWebView({ type: 'overlayTone', tone: formToneRef.current });
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
                  <Ionicons
                    name={toneMeta.icon as React.ComponentProps<typeof Ionicons>['name']}
                    size={18}
                    color={toneMeta.color}
                  />
                  <Text
                    style={[styles.feedbackText, { color: feedbackTone(metrics.feedback, colors.textSecondary) }]}
                    numberOfLines={2}
                  >
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
