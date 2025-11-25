document.addEventListener('DOMContentLoaded', () => {
    console.log('Script Loaded');

    // --- SOCKET.IO CONFIGURATION ---
    const socket = io({
        transports: ["websocket"],
        upgrade: false,
        timeout: 20000,
        reconnection: true,
    });

    // --- ELEMENTS ---
    const pages = {
        welcome: document.getElementById('welcome-page'),
        agentPlay: document.getElementById('agent-play-page'),
        overview: document.getElementById('overview-page'),
        compare: document.getElementById('compare-page')
    };

    const buttons = {
        startGame: document.getElementById('btn-start-game'),
        playVideo: document.getElementById('btn-play-video'),
        prevAction: document.getElementById('btn-prev-action'),
        nextAction: document.getElementById('btn-next-action'),
        updateAgent: document.getElementById('btn-update-agent'),
        noFeedback: document.getElementById('btn-no-feedback'),
        usePrevious: document.getElementById('btn-use-previous'),
        useUpdated: document.getElementById('btn-use-updated')
    };

    const canvases = {
        agentVideo: document.getElementById('agent-video'),
        overviewCanvas: document.getElementById('overview-canvas'),
        previousVideo: document.getElementById('previous-agent-video'),
        updatedVideo: document.getElementById('updated-agent-video')
    };

    const actionBox = document.getElementById('current-action-box');
    const actionText = document.getElementById('current-action-text');
    const actionDropdown = document.getElementById('action-dropdown');
    const totalScoreElement = document.getElementById('total-score');

    // --- STATE ---
    let currentPage = 'welcome';
    let episodeImages = [];
    let episodeActions = [];
    let currentActionIndex = 0;
    let userFeedback = [];
    let previousAgentImages = [];
    let updatedAgentImages = [];
    let totalScore = 0;

    const ACTION_NAMES = {
        0: "NOOP ⊗",
        1: "LEFT ←",
        2: "RIGHT →",
        3: "UP ↑",
        4: "DOWN ↓",
        5: "DOWN-LEFT ↙",
        6: "DOWN-RIGHT ↘",
        7: "UP-LEFT ↖",
        8: "UP-RIGHT ↗"
    };

    // --- PAGE NAVIGATION ---
    function showPage(pageName) {
        console.log('Showing page:', pageName);
        Object.values(pages).forEach(page => {
            page.classList.remove('active');
        });
        if (pages[pageName]) {
            pages[pageName].classList.add('active');
            currentPage = pageName;
        }
    }

    // --- CANVAS HELPERS ---
    function drawImageOnCanvas(canvas, base64Image) {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/png;base64,' + base64Image;
    }

    function playVideoSequence(canvas, images, fps = 10, onComplete) {
        if (!canvas || !images || images.length === 0) {
            console.error('Invalid canvas or images');
            if (onComplete) onComplete();
            return;
        }

        const ctx = canvas.getContext('2d');
        let frameIndex = 0;
        const interval = 1000 / fps;

        const playFrame = () => {
            if (frameIndex >= images.length) {
                if (onComplete) onComplete();
                return;
            }

            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                frameIndex++;
                setTimeout(playFrame, interval);
            };
            img.src = 'data:image/png;base64,' + images[frameIndex];
        };

        playFrame();
    }

    function drawActionSymbol(ctx, action, x, y, size = 30) {
        ctx.save();
        ctx.fillStyle = 'rgba(255, 255, 0, 0.8)';
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.font = `bold ${size}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        let symbol = '';
        switch(action) {
            case 1: symbol = '←'; break;
            case 2: symbol = '→'; break;
            case 3: symbol = '↑'; break;
            case 4: symbol = '↓'; break;
            case 5: symbol = '↙'; break;
            case 6: symbol = '↘'; break;
            case 7: symbol = '↖'; break;
            case 8: symbol = '↗'; break;
            case 0: symbol = '⊗'; break;
            default: symbol = '';
        }

        if (symbol) {
            ctx.strokeText(symbol, x, y);
            ctx.fillText(symbol, x, y);
        }
        ctx.restore();
    }

    // --- ACTION DROPDOWN ---
    function populateActionDropdown() {
        actionDropdown.innerHTML = '';
        Object.entries(ACTION_NAMES).forEach(([actionId, actionName]) => {
            const item = document.createElement('div');
            item.className = 'action-dropdown-item';
            item.textContent = actionName;
            item.dataset.actionId = actionId;
            item.addEventListener('click', () => {
                selectAction(parseInt(actionId));
                actionDropdown.classList.remove('show');
            });
            actionDropdown.appendChild(item);
        });
    }

    function selectAction(newActionId) {
        const originalAction = episodeActions[currentActionIndex];
        
        // Record feedback
        const existingFeedback = userFeedback.find(f => f.step === currentActionIndex);
        if (existingFeedback) {
            existingFeedback.new_action = newActionId;
        } else {
            userFeedback.push({
                step: currentActionIndex,
                original_action: originalAction,
                new_action: newActionId
            });
        }

        // Update display
        actionText.textContent = ACTION_NAMES[newActionId];
        actionText.style.background = newActionId !== originalAction ? '#ffe6e6' : '#fff';
        
        // Redraw overview with new action
        showOverviewAction(currentActionIndex);
    }

    // --- OVERVIEW PAGE LOGIC ---
    function showOverviewAction(index) {
        if (index < 0 || index >= episodeActions.length) return;
        
        currentActionIndex = index;
        const action = episodeActions[index];
        const feedback = userFeedback.find(f => f.step === index);
        const displayAction = feedback ? feedback.new_action : action;

        actionText.textContent = ACTION_NAMES[displayAction];
        actionText.style.background = feedback ? '#ffe6e6' : '#fff';

        // Draw image with action symbol
        if (episodeImages[index]) {
            drawImageOnCanvas(canvases.overviewCanvas, episodeImages[index]);
            
            // Draw action symbol overlay
            setTimeout(() => {
                const ctx = canvases.overviewCanvas.getContext('2d');
                const centerX = canvases.overviewCanvas.width / 2;
                const centerY = canvases.overviewCanvas.height / 2;
                drawActionSymbol(ctx, displayAction, centerX, centerY, 40);
            }, 100);
        }

        // Update navigation buttons
        buttons.prevAction.disabled = index === 0;
        buttons.nextAction.disabled = index === episodeActions.length - 1;
    }

    // --- EVENT LISTENERS ---
    buttons.startGame.addEventListener('click', () => {
        console.log('Start game clicked');
        socket.emit('start_game', {});
        showPage('agentPlay');
    });

    buttons.playVideo.addEventListener('click', () => {
        console.log('Play video clicked');
        buttons.playVideo.disabled = true;
        buttons.playVideo.textContent = 'Playing...';
        socket.emit('play_episode', {});
    });

    buttons.prevAction.addEventListener('click', () => {
        if (currentActionIndex > 0) {
            showOverviewAction(currentActionIndex - 1);
        }
    });

    buttons.nextAction.addEventListener('click', () => {
        if (currentActionIndex < episodeActions.length - 1) {
            showOverviewAction(currentActionIndex + 1);
        }
    });

    actionBox.addEventListener('click', (e) => {
        if (!e.target.closest('.action-dropdown')) {
            actionDropdown.classList.toggle('show');
        }
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.action-box')) {
            actionDropdown.classList.remove('show');
        }
    });

    buttons.updateAgent.addEventListener('click', () => {
        console.log('Update agent clicked', userFeedback);
        if (userFeedback.length === 0) {
            alert('No feedback provided. Please select different actions or click "No Feedback".');
            return;
        }
        socket.emit('update_agent', { feedback: userFeedback });
        showPage('compare');
    });

    buttons.noFeedback.addEventListener('click', () => {
        console.log('No feedback clicked');
        userFeedback = [];
        socket.emit('next_episode', {});
        showPage('agentPlay');
        resetAgentPlayPage();
    });

    buttons.usePrevious.addEventListener('click', () => {
        console.log('Use previous agent');
        socket.emit('select_agent', { use_updated: false });
        socket.emit('next_episode', {});
        showPage('agentPlay');
        resetAgentPlayPage();
    });

    buttons.useUpdated.addEventListener('click', () => {
        console.log('Use updated agent');
        socket.emit('select_agent', { use_updated: true });
        socket.emit('next_episode', {});
        showPage('agentPlay');
        resetAgentPlayPage();
    });

    function resetAgentPlayPage() {
        episodeImages = [];
        episodeActions = [];
        currentActionIndex = 0;
        userFeedback = [];
        totalScore = 0;
        if (totalScoreElement) totalScoreElement.textContent = '0';
        if (buttons.playVideo) {
            buttons.playVideo.disabled = false;
            buttons.playVideo.textContent = 'Play Agent';
        }
    }

    // --- SOCKET EVENTS ---
    socket.on('episode_data', (data) => {
        console.log('Received episode data:', data);
        episodeImages = data.images || [];
        episodeActions = data.actions || [];
        totalScore = data.score || 0;
        
        if (totalScoreElement) {
            totalScoreElement.textContent = totalScore.toFixed(1);
        }

        if (currentPage === 'agentPlay' && episodeImages.length > 0) {
            playVideoSequence(canvases.agentVideo, episodeImages, 10, () => {
                console.log('Episode playback complete');
                buttons.playVideo.textContent = 'Episode Complete';
                setTimeout(() => {
                    showPage('overview');
                    populateActionDropdown();
                    showOverviewAction(0);
                }, 1500);
            });
        }
    });

    socket.on('comparison_data', (data) => {
        console.log('Received comparison data:', data);
        previousAgentImages = data.previous_images || [];
        updatedAgentImages = data.updated_images || [];

        if (previousAgentImages.length > 0 && updatedAgentImages.length > 0) {
            playVideoSequence(canvases.previousVideo, previousAgentImages, 10);
            playVideoSequence(canvases.updatedVideo, updatedAgentImages, 10);
        }
    });

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });

    // Initialize
    populateActionDropdown();
    showPage('welcome');
});