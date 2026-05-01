import { useRouter } from 'expo-router';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { FeedbackBanner } from '../components/FeedbackBanner';
import { ProgressBar } from '../components/ProgressBar';
import { createMockWorkoutStream } from '../services/mock-workout-stream';
import { createLiveSessionSocket } from '../services/ws/live-session';
import { useAppState } from '../state/app-state';
import { colors } from '../theme/colors';
export function WorkoutLiveScreen() {
    const router = useRouter();
    const streamRef = useRef(null);
    const cameraRef = useRef(null);
    const wsRef = useRef(null);
    const frameTimerRef = useRef(null);
    const latestEndedSummaryRef = useRef(null);
    const [isRunning, setIsRunning] = useState(false);
    const [isMetricsExpanded, setIsMetricsExpanded] = useState(false);
    const [permission, requestPermission] = useCameraPermissions();
    const {
        authTokens,
        currentSession,
        updateMetrics,
        finishWorkout,
        latestError,
        clearError,
    } = useAppState();
    const metrics = useMemo(() => currentSession?.latestMetrics ?? {
        count: 0,
        state: 'idle',
        angle: 180,
        feedback: 'Press Start to begin.',
        progress: 0,
        formName: 'Correct',
    }, [currentSession?.latestMetrics]);
    useEffect(() => {
        return () => {
            streamRef.current?.stop();
            if (frameTimerRef.current) {
                clearInterval(frameTimerRef.current);
                frameTimerRef.current = null;
            }
            wsRef.current?.close();
        };
    }, []);

    const shouldUseMock = process.env.EXPO_PUBLIC_USE_MOCK_STREAM === 'true';

    const beginMockStreaming = () => {
        streamRef.current?.stop();
        setIsRunning(true);
        streamRef.current = createMockWorkoutStream((next) => updateMetrics(next));
    };

    const startFrameLoop = () => {
        if (frameTimerRef.current) {
            clearInterval(frameTimerRef.current);
        }
        frameTimerRef.current = setInterval(async () => {
            if (!cameraRef.current || !wsRef.current || !currentSession?.sessionId) {
                return;
            }
            try {
                const shot = await cameraRef.current.takePictureAsync({
                    quality: 0.35,
                    base64: true,
                    skipProcessing: true,
                });
                if (!shot?.base64) {
                    return;
                }
                wsRef.current.send({
                    type: 'frame',
                    sessionId: currentSession.sessionId,
                    imageJpegBase64: shot.base64,
                    timestampMs: Date.now(),
                });
            } catch {
                // Camera can fail intermittently while app is backgrounded.
            }
        }, 700);
    };

    const start = async () => {
        clearError();
        latestEndedSummaryRef.current = null;
        if (shouldUseMock || !authTokens?.accessToken || !currentSession?.sessionId) {
            beginMockStreaming();
            return;
        }
        const liveSocket = createLiveSessionSocket({
            accessToken: authTokens.accessToken,
            onStarted: () => {
                setIsRunning(true);
                startFrameLoop();
            },
            onMetrics: (payload) => {
                updateMetrics({
                    count: payload.count,
                    state: payload.state,
                    angle: Number(payload.angle ?? 180),
                    feedback: payload.feedback || 'Keep going.',
                    progress: Number(payload.progress ?? 0),
                    formName: payload.formName || 'Correct',
                });
            },
            onNoPose: () => {
                updateMetrics({
                    ...metrics,
                    feedback: 'No pose detected. Move into full camera view.',
                });
            },
            onEnded: (payload) => {
                latestEndedSummaryRef.current = payload.summary || null;
            },
            onError: () => {
                beginMockStreaming();
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
        } catch {
            wsRef.current = null;
            liveSocket.close();
            beginMockStreaming();
        }
    };

    const pause = () => {
        streamRef.current?.stop();
        setIsRunning(false);
        if (frameTimerRef.current) {
            clearInterval(frameTimerRef.current);
            frameTimerRef.current = null;
        }
    };
    const endSession = async () => {
        streamRef.current?.stop();
        setIsRunning(false);
        if (frameTimerRef.current) {
            clearInterval(frameTimerRef.current);
            frameTimerRef.current = null;
        }
        if (wsRef.current && currentSession?.sessionId) {
            wsRef.current.send({ type: 'end', sessionId: currentSession.sessionId });
        }
        await finishWorkout({ endedSummary: latestEndedSummaryRef.current });
        wsRef.current?.close();
        wsRef.current = null;
        router.replace('/(app)/session-summary');
    };
    const canShowCamera = Boolean(permission?.granted);
    return (<SafeAreaView style={styles.safe}>
      {canShowCamera ? (<CameraView ref={cameraRef} style={styles.camera} facing="front"/>) : (<View style={styles.permissionFallback}>
          <Text style={styles.permissionTitle}>Camera access needed</Text>
          <Text style={styles.permissionText}>
            Allow camera permission to start fullscreen live workout mode.
          </Text>
          <Pressable style={styles.permissionButton} onPress={() => requestPermission()}>
            <Text style={styles.permissionButtonText}>Enable camera</Text>
          </Pressable>
        </View>)}

      <View pointerEvents="box-none" style={styles.overlay}>
        <View style={styles.topHud}>
          <Text style={styles.title}>Live workout</Text>
          <Text style={styles.subtitle}>{isRunning ? 'Tracking in progress' : 'Ready to start'}</Text>
        </View>

        <View style={styles.bottomHud}>
          <View style={styles.metricsShell}>
            <Pressable style={styles.metricsToggle} onPress={() => setIsMetricsExpanded((prev) => !prev)}>
              <Text style={styles.metricsToggleText}>
                {isMetricsExpanded ? 'Hide stats' : `Show stats  Reps ${metrics.count}`}
              </Text>
              <Ionicons name={isMetricsExpanded ? 'chevron-down' : 'chevron-up'} size={16} color={colors.textPrimary}/>
            </Pressable>

            {isMetricsExpanded ? (<View style={styles.metricsCard}>
                <Text style={styles.metric}>Reps: {metrics.count}</Text>
                <Text style={styles.metric}>Angle: {metrics.angle.toFixed(1)}°</Text>
                <Text style={styles.metric}>State: {metrics.state}</Text>
                <Text style={styles.metric}>Form: {metrics.formName}</Text>
                <ProgressBar value={metrics.progress}/>
                <FeedbackBanner message={metrics.feedback}/>
                {latestError ? <Text style={styles.errorText}>{latestError}</Text> : null}
              </View>) : null}
          </View>

          <View style={styles.controls}>
            <Pressable style={styles.controlButton} onPress={start}>
              <Ionicons name={isRunning ? 'refresh' : 'play'} size={16} color={colors.textPrimary}/>
              <Text style={styles.controlText}>{isRunning ? 'Restart' : 'Play'}</Text>
            </Pressable>
            <Pressable style={[styles.controlButton, !isRunning && styles.controlButtonDisabled]} onPress={pause} disabled={!isRunning}>
              <Ionicons name="pause" size={16} color={colors.textPrimary}/>
              <Text style={styles.controlText}>Pause</Text>
            </Pressable>
            <Pressable style={[styles.controlButton, styles.controlButtonDanger]} onPress={endSession}>
              <Ionicons name="stop" size={16} color="#fff"/>
              <Text style={[styles.controlText, styles.controlTextDanger]}>End</Text>
            </Pressable>
          </View>
        </View>
      </View>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: '#000' },
    camera: { ...StyleSheet.absoluteFillObject },
    overlay: {
        flex: 1,
        paddingHorizontal: 12,
        paddingVertical: 8,
    },
    topHud: {
        backgroundColor: 'rgba(4, 10, 26, 0.55)',
        borderRadius: 12,
        paddingVertical: 10,
        paddingHorizontal: 12,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.12)',
    },
    title: { color: colors.textPrimary, fontSize: 24, fontWeight: '700' },
    subtitle: { color: colors.textSecondary, marginTop: 2 },
    metricsCard: {
        gap: 8,
        backgroundColor: 'rgba(4, 10, 26, 0.62)',
        borderRadius: 14,
        padding: 12,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.14)',
    },
    metricsShell: { gap: 8 },
    metricsToggle: {
        backgroundColor: 'rgba(4, 10, 26, 0.72)',
        borderRadius: 999,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.18)',
        paddingVertical: 7,
        paddingHorizontal: 12,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
    },
    metricsToggleText: { color: colors.textPrimary, fontSize: 12, fontWeight: '700' },
    bottomHud: {
        marginTop: 'auto',
        gap: 10,
    },
    metric: { color: colors.textPrimary, fontSize: 15 },
    controls: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 10,
        marginBottom: 4,
    },
    controlButton: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        backgroundColor: 'rgba(7, 17, 40, 0.78)',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.18)',
        paddingVertical: 8,
        paddingHorizontal: 10,
        borderRadius: 999,
    },
    controlButtonDisabled: { opacity: 0.55 },
    controlButtonDanger: {
        backgroundColor: 'rgba(220, 38, 38, 0.88)',
        borderColor: 'rgba(255,255,255,0.28)',
    },
    controlText: { color: colors.textPrimary, fontSize: 12, fontWeight: '700' },
    controlTextDanger: { color: '#fff' },
    permissionFallback: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: colors.background,
        padding: 20,
        gap: 10,
    },
    permissionTitle: { color: colors.textPrimary, fontSize: 20, fontWeight: '700' },
    permissionText: { color: colors.textSecondary, textAlign: 'center' },
    permissionButton: {
        marginTop: 8,
        backgroundColor: colors.brandAlt,
        borderRadius: 999,
        paddingVertical: 10,
        paddingHorizontal: 16,
    },
    permissionButtonText: { color: '#fff', fontWeight: '700' },
    errorText: { color: '#fca5a5', fontSize: 12 },
});
