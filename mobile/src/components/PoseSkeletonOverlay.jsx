import React, { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Circle, Line } from 'react-native-svg';
import { POSE_CONNECTIONS, POSE_VISIBILITY_THRESHOLD } from '../constants/pose-connections';

/**
 * Draws MediaPipe pose skeleton over the camera preview (normalized 0–1 coords).
 * Mirrors horizontally when using the front camera so overlay matches the preview.
 */
export function PoseSkeletonOverlay({ pose, width, height, mirror = true }) {
    const mapped = useMemo(() => {
        if (!pose?.length || !width || !height) {
            return { lines: [], dots: [] };
        }
        const toPx = (lm) => {
            const nx = mirror ? 1 - lm[0] : lm[0];
            return { x: nx * width, y: lm[1] * height, visible: lm[2] >= POSE_VISIBILITY_THRESHOLD };
        };
        const points = pose.map(toPx);
        const lines = [];
        for (const [startIdx, endIdx] of POSE_CONNECTIONS) {
            const start = points[startIdx];
            const end = points[endIdx];
            if (!start?.visible || !end?.visible) {
                continue;
            }
            lines.push({ x1: start.x, y1: start.y, x2: end.x, y2: end.y, key: `${startIdx}-${endIdx}` });
        }
        const dots = points
            .map((point, index) => (point.visible ? { ...point, key: `lm-${index}` } : null))
            .filter(Boolean);
        return { lines, dots };
    }, [pose, width, height, mirror]);

    if (!width || !height || (!mapped.lines.length && !mapped.dots.length)) {
        return null;
    }

    return (
      <View pointerEvents="none" style={StyleSheet.absoluteFill}>
        <Svg width={width} height={height}>
          {mapped.lines.map((line) => (
            <Line
              key={line.key}
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
              stroke="rgba(255,255,255,0.92)"
              strokeWidth={2.5}
              strokeLinecap="round"
            />
          ))}
          {mapped.dots.map((dot) => (
            <Circle key={dot.key} cx={dot.x} cy={dot.y} r={4} fill="#22c55e" />
          ))}
        </Svg>
      </View>
    );
}
