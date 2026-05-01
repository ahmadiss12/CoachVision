const feedbackPool = [
    'Good pace. Keep your core tight.',
    'Drive through your heels.',
    'Chest up, shoulders back.',
    'Great rep. Keep going.',
    'Control the movement on the way down.',
];
const formPool = ['Correct', 'Shallow', 'Forward Lean'];
export function createMockWorkoutStream(listener) {
    let count = 0;
    let direction = 'down';
    let progress = 0;
    const interval = setInterval(() => {
        const delta = Math.random() * 0.18 + 0.08;
        progress = direction === 'down' ? progress + delta : progress - delta;
        if (progress >= 1) {
            progress = 1;
            direction = 'up';
        }
        if (progress <= 0) {
            progress = 0;
            direction = 'down';
            count += 1;
        }
        const angle = 70 + (1 - progress) * 100;
        const feedback = feedbackPool[Math.floor(Math.random() * feedbackPool.length)];
        const formName = formPool[Math.floor(Math.random() * formPool.length)];
        const state = direction === 'down' ? 'down' : 'up';
        listener({
            count,
            state,
            angle: Number(angle.toFixed(1)),
            feedback,
            progress: Number(progress.toFixed(2)),
            formName,
        });
    }, 180);
    return {
        stop: () => clearInterval(interval),
    };
}
