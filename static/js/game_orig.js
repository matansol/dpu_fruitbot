document.addEventListener('DOMContentLoaded', () => {
    console.log('Original Game Script Loaded');

    // --- SOCKET.IO CONFIGURATION ---
    const socket = io({
        transports: ["websocket"],
        upgrade: false,
        rememberUpgrade: false,
        tryAllTransports: false,
        timeout: 20000,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        maxReconnectionAttempts: 5,
    });

    // --- ELEMENTS ---
    const coverStartButton = document.getElementById('cover-start-button');
    const welcomeContinueButton = document.getElementById('welcome-continue-button');
    const startAgentButton = document.getElementById('start-agent-button');
    const updateAgentButton = document.getElementById('update-agent-button');
    const nextEpisodeButton = document.getElementById('next-episode-button');
    const nextEpisodeSimpleButton = document.getElementById('next-episode-simple-button');
    const nextEpisodeCompareButton = document.getElementById('next-episode-compare-button');
    const useOldAgentButton = document.getElementById('use-old-agent-button');
    const prevActionButton = document.getElementById('prev-action-button');
    const nextActionButton = document.getElementById('next-action-button');
    const currentActionElement = document.getElementById('current-action');
    const actionDropdown = document.getElementById('action-dropdown');
    const loaderOverlay = document.getElementById('loader-overlay');
    const ph2PlaceholderSpinner = document.getElementById('ph2-placeholder-spinner');
    const gameImagePh2 = document.getElementById('game-image-ph2');
    const overviewGameImage = document.getElementById('overview-game-image');
    const feedbackExplanationInput = document.getElementById('feedback-explanation-input');

    // --- STATE ---
    let currentPhase = 'cover';
    let ph2ImageLoaded = false;
    let phase2_counter = 1;
    let actions = [];
    let currentActionIndex = 0;
    let feedbackImages = [];
    let cumulative_rewards = [];
    let userFeedback = [];
    let changedIndexes = [];
    let actionsCells = [];
    let agentGroup = null;

    const actionsNameList = ['move forward', 'turn right', 'turn left', 'pickup'];

    // --- PROLIFIC ID HANDLING ---
    function getProlificIdOrRandom() {
        const params = new URLSearchParams(window.location.search);
        let prolificId = params.get('prolificID');
        if (prolificId && prolificId.trim() !== '') {
            return prolificId;
        } else {
            return Math.floor(Math.random() * 100) + 1;
        }
    }
    const prolificID = getProlificIdOrRandom();
    console.log("Prolific ID:", prolificID);

    // --- PAGE HELPERS ---
    function showPage(pageId) {
        console.log('showPage', pageId);

        if (pageId === 'ph2-game-page') {
            if (phase2_counter > 5) {
                showPage('summary-page');
                return;
            }
            ph2ImageLoaded = false;
            if (ph2PlaceholderSpinner) ph2PlaceholderSpinner.style.display = 'block';
            if (gameImagePh2) gameImagePh2.style.visibility = 'hidden';

            if (startAgentButton) {
                startAgentButton.disabled = false;
                startAgentButton.style.backgroundColor = '';
                startAgentButton.style.color = '';
            }

            const roundNumberElem = document.getElementById('round-number');
            if (roundNumberElem) {
                roundNumberElem.textContent = phase2_counter;
            }
        }

        if (pageId === 'summary-page') {
            const confirmationCodeElement = document.getElementById('confirmation-code');
            if (confirmationCodeElement) {
                const finalAgentGroup = agentGroup !== null ? agentGroup : 1;
                confirmationCodeElement.textContent = `APPL${finalAgentGroup}`;
            }
        }

        document.querySelectorAll('.page').forEach(p => {
            p.classList.remove('active');
            p.style.display = 'none';
        });

        const page = document.getElementById(pageId);
        if (!page) return console.error(`Page "${pageId}" not found.`);
        page.classList.add('active');
        page.style.display = 'flex';

        if (pageId === 'overview-page') {
            const overviewRoundElem = document.getElementById('overview-round-number');
            if (overviewRoundElem) {
                overviewRoundElem.textContent = phase2_counter - 1;
            }
        }
    }

    function showLoader() { if (loaderOverlay) loaderOverlay.style.display = 'flex'; }
    function hideLoader() { if (loaderOverlay) loaderOverlay.style.display = 'none'; }

    // --- INITIAL PAGE ---
    showPage('cover-page');

    // --- NAVIGATION ---
    if (coverStartButton) {
        coverStartButton.addEventListener('click', () => {
            showPage('welcome-page');
        });
    }

    if (welcomeContinueButton) {
        welcomeContinueButton.addEventListener('click', () => {
            socket.emit('start_game', { playerName: prolificID, updateAgent: false });
            showPage('ph2-game-page');
        });
    }

    // --- PHASE 2 IMAGE HANDLING ---
    if (gameImagePh2) {
        gameImagePh2.onload = function () {
            if (!ph2ImageLoaded) {
                if (ph2PlaceholderSpinner) ph2PlaceholderSpinner.style.display = 'none';
                gameImagePh2.style.visibility = 'visible';
                ph2ImageLoaded = true;
            }
        };
    }

    function setPh2GameImage(src) {
        if (!ph2ImageLoaded) {
            if (ph2PlaceholderSpinner) ph2PlaceholderSpinner.style.display = 'block';
            if (gameImagePh2) gameImagePh2.style.visibility = 'hidden';
        }
        if (gameImagePh2) gameImagePh2.src = src;
    }

    // --- EPISODE NAVIGATION ---
    if (nextEpisodeButton) {
        nextEpisodeButton.addEventListener('click', () => {
            const feedbackExplanation = feedbackExplanationInput ? feedbackExplanationInput.value : "";
            socket.emit('start_game', { playerName: prolificID, updateAgent: false, userNoFeedback: true, userExplanation: feedbackExplanation });
            resetOverviewHighlights();
            showPage('ph2-game-page');
        });
    }

    if (nextEpisodeSimpleButton) {
        nextEpisodeSimpleButton.addEventListener('click', () => {
            resetOverviewHighlights();
            socket.emit('start_game', { playerName: prolificID, updateAgent: false });
            showPage('ph2-game-page');
        });
    }

    if (nextEpisodeCompareButton) {
        nextEpisodeCompareButton.addEventListener('click', () => {
            socket.emit('agent_selected', { use_old_agent: false });
            socket.emit('start_game', { playerName: prolificID, updateAgent: false });
            showPage('ph2-game-page');
        });
    }

    if (useOldAgentButton) {
        useOldAgentButton.addEventListener('click', () => {
            socket.emit('agent_selected', { use_old_agent: true });
            socket.emit('start_game', { playerName: prolificID, updateAgent: false });
            showPage('ph2-game-page');
        });
    }

    // --- AGENT BUTTONS ---
    if (startAgentButton) {
        startAgentButton.addEventListener('click', () => {
            startAgentButton.disabled = true;
            startAgentButton.style.backgroundColor = '#a9a9a9';
            startAgentButton.style.color = '#fff';
            socket.emit('play_entire_episode');
        });
    }

    // --- UPDATE AGENTS BUTTON ---
    if (updateAgentButton) {
        updateAgentButton.addEventListener('click', () => {
            const feedbackExplanation = feedbackExplanationInput ? feedbackExplanationInput.value : "";
            if (feedbackExplanationInput) feedbackExplanationInput.value = '';
            
            if (userFeedback.length === 0) {
                alert('No feedback was given. You cannot update the agent without providing feedback.');
                return;
            }
            
            showLoader();
            socket.emit('compare_agents', {
                playerName: prolificID,
                updateAgent: true,
                userFeedback,
                feedbackExplanationText: feedbackExplanation,
            });
            userFeedback = [];
            resetOverviewHighlights();
            
            setTimeout(() => {
                hideLoader();
                showPage('compare-agent-update-page');
            }, 4000);
        });
    }

    // --- SOCKET EVENTS ---
    socket.on('game_update', data => {
        const activePage = document.querySelector('.page.active');
        if (!activePage) return;
        
        if (activePage.id === 'ph2-game-page') {
            setPh2GameImage('data:image/png;base64,' + data.image);
            const rewardElem = document.getElementById('reward2');
            if (rewardElem) rewardElem.innerText = data.reward;
            const scoreElem = document.getElementById('score2');
            if (scoreElem) scoreElem.innerText = data.score;
            const stepCountElementPh2 = document.getElementById('step-count-ph2');
            if (stepCountElementPh2) stepCountElementPh2.innerText = data.step_count;
        }
    });

    socket.on('episode_finished', data => {
        phase2_counter += 1;
        const roundNumberElem = document.getElementById('round-number');
        if (roundNumberElem) {
            roundNumberElem.textContent = phase2_counter;
        }
        updateOverviewPage(data);
    });

    socket.on('compare_agents', data => {
        let rawImageSrc = data.rawImage;
        if (rawImageSrc && !rawImageSrc.startsWith("data:image")) {
            rawImageSrc = "data:image/png;base64," + rawImageSrc;
        }
        drawPathOnCanvas('previous-agent-canvas', rawImageSrc, data.prevMoveSequence);
        drawPathOnCanvas('updated-agent-canvas', rawImageSrc, data.updatedMoveSequence);
    });

    socket.on('update_agent_group', data => {
        if (data && data.agent_group) {
            agentGroup = data.agent_group;
            console.log("Updated agentGroup to", agentGroup);
        }
    });

    // --- OVERVIEW PAGE LOGIC ---
    function updateOverviewPage(data) {
        actions = data.actions.map(a => ({ ...a, orig_action: a.action }));
        currentActionIndex = 0;
        feedbackImages = data.feedback_images;
        actionsCells = data.actions_cells;
        cumulative_rewards = data.cumulative_rewards;
        
        showCurrentAction();
        showPage('overview-page');
    }

    function showCurrentAction() {
        if (actions.length === 0) return;
        
        const currentAction = actions[currentActionIndex];
        if (overviewGameImage) {
            overviewGameImage.src = 'data:image/png;base64,' + feedbackImages[currentActionIndex];
        }
        
        if (currentActionElement) {
            currentActionElement.textContent = currentAction.action;
            currentActionElement.classList.remove('selected-action');
            if (changedIndexes.includes(currentActionIndex)) {
                currentActionElement.classList.add('selected-action');
            }
        }
        
        if (nextActionButton) {
            nextActionButton.style.display = currentActionIndex >= actions.length - 1 ? 'none' : 'inline-block';
        }

        // Update reward and score
        const reward2Elem = document.getElementById('reward2');
        const score2Elem = document.getElementById('score2');
        if (Array.isArray(cumulative_rewards) && cumulative_rewards.length > 0) {
            const idx = currentActionIndex;
            let reward = 0;
            let score = cumulative_rewards[idx];
            if (idx === 0) {
                reward = cumulative_rewards[0];
            } else {
                reward = cumulative_rewards[idx] - cumulative_rewards[idx - 1];
            }
            reward = Math.round(reward * 10) / 10;
            score = Math.round(score * 10) / 10;
            if (reward2Elem) reward2Elem.textContent = reward;
            if (score2Elem) score2Elem.textContent = score;
        }
    }

    if (prevActionButton) {
        prevActionButton.addEventListener('click', () => {
            if (currentActionIndex > 0) {
                currentActionIndex--;
                showCurrentAction();
            }
        });
    }

    if (nextActionButton) {
        nextActionButton.addEventListener('click', () => {
            if (currentActionIndex < actions.length - 1) {
                currentActionIndex++;
                showCurrentAction();
            }
        });
    }

    if (currentActionElement) {
        currentActionElement.addEventListener('click', () => {
            if (actionDropdown) {
                const rect = currentActionElement.getBoundingClientRect();
                actionDropdown.style.left = `${rect.left}px`;
                actionDropdown.style.top = `${rect.bottom + window.scrollY}px`;
                actionDropdown.style.display = 'block';
            }
        });
    }

    document.addEventListener('click', event => {
        if (!event.target.closest('#current-action') && !event.target.closest('.dropdown')) {
            if (actionDropdown) actionDropdown.style.display = 'none';
        }
    });

    // Populate dropdown
    if (actionDropdown) {
        actionDropdown.querySelectorAll('.dropdown-item').forEach((item, index) => {
            item.addEventListener('click', () => {
                handleActionSelection(currentActionIndex, actionsNameList[index]);
                actionDropdown.style.display = 'none';
            });
        });
    }

    function handleActionSelection(index, newAction) {
        const originalAction = actions[index].orig_action;
        actions[index].action = newAction;
        userFeedback.push({ index, agent_action: originalAction, feedback_action: newAction });
        
        const idx = changedIndexes.indexOf(index);
        if (newAction !== originalAction) {
            if (idx === -1) changedIndexes.push(index);
        } else {
            if (idx !== -1) changedIndexes.splice(idx, 1);
        }
        
        showCurrentAction();
    }

    function resetOverviewHighlights() {
        changedIndexes = [];
        if (currentActionElement) currentActionElement.classList.remove('selected-action');
    }

    // --- DRAWING HELPERS ---
    function drawPathOnCanvas(canvasId, imageSrc, moveSequence) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = function () {
            canvas.width = img.width * 2;
            canvas.height = img.height * 2;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Draw path visualization here
            // This is a simplified version - implement based on your needs
        };
        
        img.src = imageSrc;
    }

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });
});
