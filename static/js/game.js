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
        compare: document.getElementById('compare-page'),
        agentUpdated: document.getElementById('agent-updated-page'),
        finish: document.getElementById('finish-page')
    };

    // DEBUG: Check if all pages were found
    console.log('[DEBUG] Page elements loaded:', {
        welcome: !!pages.welcome,
        agentPlay: !!pages.agentPlay,
        overview: !!pages.overview,
        compare: !!pages.compare,
        agentUpdated: !!pages.agentUpdated,
        finish: !!pages.finish
    });

    const buttons = {
        startGame: document.getElementById('btn-start-game'),
        playVideo: document.getElementById('btn-play-video'),
        playSequence: document.getElementById('btn-play-sequence'),
        playBackward: document.getElementById('btn-play-backward'),
        pauseSequence: document.getElementById('btn-pause-sequence'),
        prevAction: document.getElementById('btn-prev-action'),
        nextAction: document.getElementById('btn-next-action'),
        updateAgent: document.getElementById('btn-update-agent'),
        noFeedback: document.getElementById('btn-no-feedback'),
        usePrevious: document.getElementById('btn-use-previous'),
        useUpdated: document.getElementById('btn-use-updated'),
        continueNextEpisode: document.getElementById('btn-continue-next-episode')
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
    const loadingOverlay = document.getElementById('loading-overlay');
    const previousAgentImage = document.getElementById('previous-agent-image');
    const updatedAgentImage = document.getElementById('updated-agent-image');

    // --- GET PLAYER NAME FROM URL OR GENERATE RANDOM ---
    function getPlayerNameFromURL() {
        const urlParams = new URLSearchParams(window.location.search);
        // Try both 'prolificId' and 'prolificID' (case variations)
        const prolificId = urlParams.get('prolificId') || urlParams.get('prolificID');

        if (prolificId) {
            console.log('Prolific ID:', prolificId);
            return prolificId;
        }

        // Generate random number between 1 and 100
        const randomId = Math.floor(Math.random() * 100) + 1;
        console.log('Generated random player ID:', randomId);
        return randomId.toString();
    }
    // --- GET PLAYER Group FROM URL OR GENERATE RANDOM ---
    function getPlayerGroupFromURL() {
        const urlParams = new URLSearchParams(window.location.search);
        console.log('DEBUG: Full URL:', window.location.href);
        console.log('DEBUG: Search string:', window.location.search);
        console.log('DEBUG: All URL params:', Array.from(urlParams.entries()));
        const group = urlParams.get('group');
        console.log('DEBUG: Group value from URL:', group, 'Type:', typeof group);

        if (group) {
            const groupInt = parseInt(group, 10);
            console.log('Group:', groupInt);
            return groupInt;
        }

        console.log('could not find group, return default value=1');
        return 1;
    }

    // --- STATE ---
    let currentPage = 'welcome';
    let episodeImages = [];
    let episodeActions = [];
    let episodeRewards = [];  // Add rewards array
    let episodePositions = [];  // Add positions array
    let episodeCollisions = [];  // Add collisions array
    let currentActionIndex = 0;
    let userFeedback = [];
    let previousAgentImages = [];
    let updatedAgentImages = [];
    let totalScore = 0;
    let isPlayingVideo = false;  // Track if video playback is active
    let playerName = getPlayerNameFromURL();
    let group = getPlayerGroupFromURL();
    let episodeCount = 0;  // Track number of episodes completed
    const MAX_EPISODES = 4;  // End game after 4 episodes

    // Playback state
    let isPlaying = false;
    let playbackInterval = null;

    const ACTION_NAMES = {
        0: "LEFT ←",
        1: "UP",
        2: "RIGHT →",
        3: "THROW",
    };

    // --- PAGE NAVIGATION ---
    function showPage(pageName) {
        console.log('[DEBUG] showPage called with:', pageName);

        Object.values(pages).forEach(page => {
            if (page) page.classList.remove('active');
        });

        if (pages[pageName]) {
            console.log(`[DEBUG] Activating page: ${pageName}`);
            pages[pageName].classList.add('active');
            currentPage = pageName;
        } else {
            console.error(`[DEBUG] ERROR: Page '${pageName}' not found in pages object!`);
        }
    }

    // --- CANVAS HELPERS ---
    function drawImageOnCanvas(canvas, base64Image) {

        if (!canvas) {
            console.error('[drawImageOnCanvas] Canvas is null or undefined');
            return;
        }

        if (!base64Image) {
            console.error('[drawImageOnCanvas] base64Image is null or undefined');
            return;
        }

        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };

        img.onerror = (error) => {
            console.error('[drawImageOnCanvas] Image failed to load:', error);
            console.error('[drawImageOnCanvas] Image src length:', img.src?.length);
            console.error('[drawImageOnCanvas] Image src prefix:', img.src?.substring(0, 100));
        };

        img.src = 'data:image/jpeg;base64,' + base64Image;
    }

    function playVideoSequence(canvas, images, fps = 10, onComplete, onFrameUpdate = null) {
        if (!canvas || !images || images.length === 0) {
            console.error('[playVideoSequence] Invalid canvas or images:', {
                canvasExists: !!canvas,
                imagesExists: !!images,
                imagesLength: images?.length
            });
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
                
                // Call frame update callback with current frame index
                if (onFrameUpdate) {
                    onFrameUpdate(frameIndex, images.length);
                }
                
                frameIndex++;
                setTimeout(playFrame, interval);
            };

            img.onerror = (error) => {
                console.error(`[playVideoSequence] Frame ${frameIndex} failed to load:`, error);
                frameIndex++;
                setTimeout(playFrame, interval);
            };

            img.src = 'data:image/jpeg;base64,' + images[frameIndex];
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
        switch (action) {
            case 0: symbol = '←'; break;
            case 1: symbol = '↑'; break;
            case 2: symbol = '→'; break;
            // case 3: symbol = '!'; break;  // Throw - no symbol
            default: symbol = '';
        }

        if (symbol) {
            ctx.strokeText(symbol, x, y);
            ctx.fillText(symbol, x, y);
        }
        ctx.restore();
    }

    function worldToPixel(worldX, worldY, agentY, canvasWidth, canvasHeight, mainWidth = 10, mainHeight = 15, res = 64) {
        // FruitBot camera follows agent vertically
        const visibility = mainWidth;

        // Camera center calculation (from choose_center in fruitbot.cpp)
        const centerX = mainWidth / 2.0;
        const centerY = agentY + mainWidth / 2.0 - 2 * 0.25;  // agent->ry is typically 0.25

        // Calculate unit scale (from prepare_for_drawing)
        const rawUnit = res / visibility;
        const viewDim = res / rawUnit;

        // Calculate offsets
        const xOff = rawUnit * (centerX - viewDim / 2);
        const yOff = rawUnit * (centerY - viewDim / 2);

        // Convert world to screen coordinates
        // Note: Y is inverted (viewDim - y) because screen Y increases downward
        let pixelX = worldX * rawUnit - xOff;
        let pixelY = (viewDim - worldY) * rawUnit + yOff;

        // Scale from 64x64 to canvas size
        pixelX = (pixelX / res) * canvasWidth;
        pixelY = (pixelY / res) * canvasHeight;

        return { x: pixelX, y: pixelY };
    }

    function drawCollisionMarkers(ctx, collisions, currentStep) {
        if (!collisions || collisions.length === 0) return;

        const canvasWidth = ctx.canvas.width;
        const canvasHeight = ctx.canvas.height;

        ctx.save();

        // Draw X marks for all collisions up to current step
        collisions.forEach(collision => {
            if (collision.step <= currentStep) {
                // Convert world coordinates to pixel coordinates
                const pos = worldToPixel(
                    collision.world_x,
                    collision.world_y,
                    collision.agent_y,
                    canvasWidth,
                    canvasHeight
                );
                const x = pos.x;
                const y = pos.y;

                const size = 8;  // Size of the X mark

                // Color based on collision type
                if (collision.type === 7) {
                    // GOOD_OBJ (fruit) - green X
                    ctx.strokeStyle = 'rgba(251, 255, 0, 0.9)';
                    ctx.lineWidth = 3;
                } else if (collision.type === 4) {
                    // BAD_OBJ (vegetable) - red X
                    ctx.strokeStyle = 'rgba(251, 255, 0, 0.9)';
                    ctx.lineWidth = 3;
                } else if (collision.type === 1 || collision.type === 10) {
                    // BARRIER or LOCKED_DOOR - orange X
                    ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
                    ctx.lineWidth = 3;
                } else {
                    // Other collisions - white X
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                    ctx.lineWidth = 2;
                }

                // Draw X mark
                ctx.beginPath();
                ctx.moveTo(x - size, y - size);
                ctx.lineTo(x + size, y + size);
                ctx.moveTo(x + size, y - size);
                ctx.lineTo(x - size, y + size);
                ctx.stroke();
            }
        });

        ctx.restore();
    }

    // --- ACTION DROPDOWN ---
    function populateActionDropdown() {
        actionDropdown.innerHTML = '';
        const currentAgentAction = episodeActions[currentActionIndex];
        const feedback = userFeedback.find(f => f.index === currentActionIndex);
        const userSelectedAction = feedback ? feedback.feedback_action : null;

        console.log('[populateActionDropdown] Current agent action:', currentAgentAction, 'User selected:', userSelectedAction);

        Object.entries(ACTION_NAMES).forEach(([actionId, actionName]) => {
            const item = document.createElement('div');
            item.className = 'action-dropdown-item';
            const actionIdNum = parseInt(actionId);

            // Mark current agent action
            if (actionIdNum === currentAgentAction) {
                item.classList.add('current-agent-action');
                item.textContent = actionName + ' (Current)';
            } else {
                item.textContent = actionName;
            }

            // Mark user-selected action with light blue background
            if (userSelectedAction !== null && actionIdNum === userSelectedAction) {
                item.classList.add('user-selected');
            }

            item.dataset.actionId = actionId;
            item.addEventListener('click', () => {
                selectAction(actionIdNum);
                actionDropdown.classList.remove('show');
            });
            actionDropdown.appendChild(item);
        });
    }

    function selectAction(newActionId) {
        const originalAction = episodeActions[currentActionIndex];

        // Record feedback
        const existingFeedback = userFeedback.find(f => f.index === currentActionIndex);
        if (existingFeedback) {
            existingFeedback.feedback_action = newActionId;
        } else {
            userFeedback.push({
                index: currentActionIndex,
                agent_action: originalAction,
                feedback_action: newActionId
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
        const feedback = userFeedback.find(f => f.index === index);
        const displayAction = feedback ? feedback.feedback_action : action;

        // Get action name with fallback for undefined actions
        const actionName = ACTION_NAMES[displayAction] || `Action ${displayAction}`;
        actionText.textContent = actionName;
        actionText.style.background = feedback ? '#ffe6e6' : '#fff';

        // Draw image with action symbol
        if (episodeImages[index]) {
            drawImageOnCanvas(canvases.overviewCanvas, episodeImages[index]);

            // Draw action symbol overlay at bot position
            setTimeout(() => {
                const ctx = canvases.overviewCanvas.getContext('2d');
                const canvasHeight = canvases.overviewCanvas.height;
                const canvasWidth = canvases.overviewCanvas.width;

                // Position symbol at bottom 1/10 of canvas
                const symbolY = canvasHeight - (canvasHeight / 10);

                // Calculate position: use episodePositions[0] as starting point, then track movements
                let symbolX;
                if (index === 0) {
                    // First frame: use position from data or canvas center
                    symbolX = episodePositions[0] + 20 !== undefined ? episodePositions[0] : canvasWidth / 2;
                } else {
                    // Subsequent frames: calculate from starting position + accumulated actions
                    const startX = episodePositions[0] + 20 !== undefined ? episodePositions[0] : canvasWidth / 2;
                    const moveSize = 35; // Same as Python backend
                    const margin = 20; // Keep arrow within bounds
                    
                    // Calculate position based on all actions up to current index
                    symbolX = startX;
                    for (let i = 0; i < index; i++) {
                        const action = episodeActions[i];
                        if (action === 0) { // LEFT
                            symbolX = Math.max(margin, symbolX - moveSize);
                        } else if (action === 2) { // RIGHT
                            symbolX = Math.min(canvasWidth - margin, symbolX + moveSize);
                        }
                        // UP (1) and THROW (3) don't change position
                    }
                }

                const margin = 20; // pixels from edge
                const shiftAmount = 35; // pixels to shift for left/right arrows
                
                // Clamp the base position within bounds
                symbolX = Math.max(margin, Math.min(symbolX, canvasWidth - margin));
                
                // Then shift arrow based on action direction, ensuring it stays in bounds
                if (displayAction === 0) { // LEFT
                    symbolX = Math.max(margin, symbolX - shiftAmount);
                } else if (displayAction === 2) { // RIGHT
                    symbolX = Math.min(canvasWidth - margin, symbolX + shiftAmount);
                }

                drawActionSymbol(ctx, displayAction, symbolX, symbolY, 40);

                // Draw collision markers for all collisions up to current step
                // drawCollisionMarkers(ctx, episodeCollisions, index);
            }, 100);
        }

        // Update navigation buttons - always enabled for cycling
        buttons.prevAction.disabled = false;
        buttons.nextAction.disabled = false;

        // Refresh dropdown to show current selections
        populateActionDropdown();
    }

    // --- EVENT LISTENERS ---
    buttons.startGame.addEventListener('click', () => {
        console.log('[startGame] Player name:', playerName);
        console.log('[startGame] Player group:', group);
        console.log('[startGame] Emitting start_game event');
        socket.emit('start_game', { playerName: playerName, group: group });
        showPage('agentPlay');
        console.log('[startGame] Switched to agentPlay page');
    });

    buttons.playVideo.addEventListener('click', () => {
        console.log('[playVideo] Button clicked');
        buttons.playVideo.disabled = true;
        buttons.playVideo.textContent = 'Playing...';
        console.log('[playVideo] Emitting play_episode event');
        socket.emit('play_episode', {});
    });

    buttons.playSequence.addEventListener('click', () => {
        if (episodeActions.length === 0) return;

        // If at the end, restart from beginning
        if (currentActionIndex >= episodeActions.length - 1) {
            currentActionIndex = -1; // Will increment to 0 in first iteration
        }

        isPlaying = true;
        buttons.playSequence.style.display = 'none';
        buttons.pauseSequence.style.display = 'flex';

        playbackInterval = setInterval(() => {
            if (currentActionIndex < episodeActions.length - 1) {
                showOverviewAction(currentActionIndex + 1);
            } else {
                // Reached end, stop playing
                stopPlayback();
            }
        }, 100); // 0.1 seconds per frame
    });

    // Backward button removed - not needed in feedback interface

    buttons.pauseSequence.addEventListener('click', () => {
        stopPlayback();
    });

    function stopPlayback() {
        isPlaying = false;
        if (playbackInterval) {
            clearInterval(playbackInterval);
            playbackInterval = null;
        }
        buttons.playSequence.style.display = 'flex';
        buttons.pauseSequence.style.display = 'none';
    }

    buttons.prevAction.addEventListener('click', () => {
        stopPlayback(); // Stop auto-play when manually navigating
        if (currentActionIndex > 0) {
            showOverviewAction(currentActionIndex - 1);
        } else {
            // At beginning, wrap to end
            showOverviewAction(episodeActions.length - 1);
        }
    });

    buttons.nextAction.addEventListener('click', () => {
        stopPlayback(); // Stop auto-play when manually navigating
        if (currentActionIndex < episodeActions.length - 1) {
            showOverviewAction(currentActionIndex + 1);
        } else {
            // At end, restart from beginning
            showOverviewAction(0);
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

        // Show loading screen
        showLoading();
        console.log('[updateAgent] Emitting compare_agents event');

        socket.emit('compare_agents', {
            playerName: playerName,
            updateAgent: true,
            userFeedback: userFeedback
        });
    });

    buttons.noFeedback.addEventListener('click', () => {
        console.log('No feedback clicked');
        userFeedback = [];
        episodeCount++;
        console.log(`[noFeedback] Episode ${episodeCount} of ${MAX_EPISODES} completed`);
        socket.emit('next_episode', { playerName: playerName });
        showPage('agentPlay');
        resetAgentPlayPage();
    });

    buttons.usePrevious.addEventListener('click', () => {
        console.log('Use previous agent');
        socket.emit('agent_select', {
            playerName: playerName,
            use_updated: false
        });
        episodeCount++;
        console.log(`[usePrevious] Episode ${episodeCount} of ${MAX_EPISODES} completed`);
        socket.emit('next_episode', { playerName: playerName });
        showPage('agentPlay');
        resetAgentPlayPage();
    });

    buttons.useUpdated.addEventListener('click', () => {
        console.log('Use updated agent');
        socket.emit('agent_select', {
            playerName: playerName,
            use_updated: true
        });
        episodeCount++;
        console.log(`[useUpdated] Episode ${episodeCount} of ${MAX_EPISODES} completed`);
        socket.emit('next_episode', { playerName: playerName });
        showPage('agentPlay');
        resetAgentPlayPage();
    });

    // Button handler for similarity level 0 - continue to next episode after agent update
    buttons.continueNextEpisode.addEventListener('click', () => {
        console.log('[continueNextEpisode] Continuing to next episode after agent update (similarity level 0)');
        episodeCount++;
        console.log(`[continueNextEpisode] Episode ${episodeCount} of ${MAX_EPISODES} completed`);
        socket.emit('next_episode', { playerName: playerName });
        showPage('agentPlay');
        resetAgentPlayPage();
    });

    function resetAgentPlayPage() {
        episodeImages = [];
        episodeActions = [];
        episodeRewards = [];  // Reset rewards
        episodePositions = [];  // Reset positions
        episodeCollisions = [];  // Reset collisions
        currentActionIndex = 0;
        userFeedback = [];
        totalScore = 0;
        isPlayingVideo = false;
        if (totalScoreElement) totalScoreElement.textContent = '0';
        if (buttons.playVideo) {
            buttons.playVideo.disabled = false;
            buttons.playVideo.textContent = 'Play Agent';
        }
    }

    function showLoading() {
        console.log('[showLoading] Showing loading overlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('show');
        }
    }

    function hideLoading() {
        console.log('[hideLoading] Hiding loading overlay');
        if (loadingOverlay) {
            loadingOverlay.classList.remove('show');
        }
    }

    // --- SOCKET EVENTS ---
    socket.on('game_update', (data) => {
        console.log('[game_update] ===== RECEIVED =====');
        console.log('[game_update] Data keys:', Object.keys(data));
        console.log('[game_update] Full data:', {
            episode: data.episode,
            score: data.score,
            done: data.done,
            reward: data.reward,
            agent_action: data.agent_action,
            action: data.action,
            imageExists: !!data.image,
            imageLength: data.image?.length,
            imagePrefix: data.image?.substring(0, 50)
        });

        if (data.image) {
            console.log('[game_update] Image data present, length:', data.image.length);
            console.log('[game_update] Calling drawImageOnCanvas for agent-video');
            console.log('[game_update] Canvas element:', canvases.agentVideo);
            console.log('[game_update] Canvas dimensions before draw:', {
                width: canvases.agentVideo?.width,
                height: canvases.agentVideo?.height,
                offsetWidth: canvases.agentVideo?.offsetWidth,
                offsetHeight: canvases.agentVideo?.offsetHeight
            });

            drawImageOnCanvas(canvases.agentVideo, data.image);
        } else {
            console.error('[game_update] NO IMAGE DATA in response!');
        }

        if (data.score !== undefined && totalScoreElement) {
            console.log('[game_update] Updating score to:', data.score);
            totalScoreElement.textContent = data.score;
        }

        console.log('[game_update] ===== END =====');
    });

    // --- NEW: Handle batched streaming of episode frames ---
    socket.on('episode_batch', (data) => {
        console.log('[episode_batch] ===== RECEIVED BATCH =====');
        console.log('[episode_batch] Batch data:', {
            batchStart: data.batch_start,
            newFrames: data.images?.length,
            totalFramesNow: episodeImages.length + (data.images?.length || 0),
            isFinal: data.is_final,
            score: data.score
        });

        // Append new batch data to existing arrays
        if (data.images) episodeImages.push(...data.images);
        if (data.actions) episodeActions.push(...data.actions);
        if (data.rewards) episodeRewards.push(...data.rewards);
        if (data.positions) episodePositions.push(...data.positions);
        if (data.collisions) episodeCollisions = data.collisions;  // Replace with latest
        if (data.score !== undefined) totalScore = data.score;

        console.log('[episode_batch] Actions with names:', episodeActions.map((a, i) => `${i}: ${a} (${ACTION_NAMES[a] || 'UNKNOWN'})`));

        if (totalScoreElement) {
            totalScoreElement.textContent = totalScore.toFixed(1);
        }

        // If this is the final batch, start playback
        if (data.is_final && currentPage === 'agentPlay' && episodeImages.length > 0) {
            console.log('[episode_batch] Final batch received, starting playback with', episodeImages.length, 'frames');
            isPlayingVideo = true;
            playVideoSequence(canvases.agentVideo, episodeImages, 10, 
                () => {
                    console.log('[episode_batch] Episode playback complete');
                    isPlayingVideo = false;
                    buttons.playVideo.textContent = 'Episode Complete';
                    setTimeout(() => {
                        showPage('overview');
                        populateActionDropdown();
                        showOverviewAction(0);
                    }, 1500);
                },
                (frameIndex, totalFrames) => {
                    // Update score using cumulative rewards up to current frame
                    let currentScore = 0;
                    for (let i = 0; i <= frameIndex && i < episodeRewards.length; i++) {
                        currentScore += episodeRewards[i];
                    }
                    if (totalScoreElement) {
                        totalScoreElement.textContent = currentScore.toFixed(1);
                    }
                }
            );
        } else if (data.is_final) {
            console.log('[episode_batch] Final batch received');
        } else {
            console.log('[episode_batch] Batch accumulated, waiting for more...');
        }

        console.log('[episode_batch] ===== END =====');
    });

    socket.on('episode_data', (data) => {
        console.log('[episode_data] ===== RECEIVED =====');
        console.log('[episode_data] Actions with names:', data.actions?.map((a, i) => `${i}: ${a} (${ACTION_NAMES[a] || 'UNKNOWN'})`));

        // Only use episode_data if we haven't received batches
        // (backwards compatibility or fallback)
        if (episodeImages.length === 0) {
            episodeImages = data.images || [];
            episodeActions = data.actions || [];
            episodePositions = data.positions || [];
            episodeCollisions = data.collisions || [];  // Add collisions
            totalScore = data.score || 0;

            if (totalScoreElement) {
                totalScoreElement.textContent = totalScore.toFixed(1);
            }

            if (currentPage === 'agentPlay' && episodeImages.length > 0) {
                console.log('[episode_data] Playing video sequence with', episodeImages.length, 'frames');
                const finalScore = totalScore;
                playVideoSequence(canvases.agentVideo, episodeImages, 10, 
                    () => {
                        console.log('[episode_data] Episode playback complete');
                        buttons.playVideo.textContent = 'Episode Complete';
                        setTimeout(() => {
                            showPage('overview');
                            populateActionDropdown();
                            showOverviewAction(0);
                        }, 1500);
                    },
                    (frameIndex, totalFrames) => {
                        // Update score progressively during playback
                        const progress = frameIndex / totalFrames;
                        const currentScore = progress * finalScore;
                        if (totalScoreElement) {
                            totalScoreElement.textContent = currentScore.toFixed(1);
                        }
                    }
                );
            } else {
                console.warn('[episode_data] Not playing video:', {
                    currentPage,
                    imagesLength: episodeImages.length
                });
            }
        } else {
            console.log('[episode_data] Ignoring - already received batches');
        }
        console.log('[episode_data] ===== END =====');
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
        console.log('[socket] ===== CONNECTED TO SERVER =====');
        console.log('[socket] Socket ID:', socket.id);
    });

    socket.on('disconnect', () => {
        console.log('[socket] ===== DISCONNECTED FROM SERVER =====');
    });

    socket.on('error', (data) => {
        console.error('[socket] ===== ERROR FROM SERVER =====');
        console.error('[socket] Error data:', data);
        alert(`Server error: ${data.message || data.error || 'Unknown error'}`);
    });

    socket.on('connect_error', (error) => {
        console.error('[socket] ===== CONNECTION ERROR =====');
        console.error('[socket] Error:', error);
    });

    socket.on('compare_agents', (data) => {
        console.log('[compare_agents] ===== RECEIVED =====');
        console.log('[compare_agents] Data keys:', Object.keys(data));
        console.log('[compare_agents] Similarity level:', data.similarity_level);
        console.log('[compare_agents] Agent updated:', data.agent_updated);
        console.log('[compare_agents] Has rawImageUpdate:', !!data.rawImageUpdated);
        console.log('[compare_agents] Has rawImagePrev:', !!data.rawImagePrev);

        // Hide loading screen
        hideLoading();

        // Check if this is a similarity level 0 response (no comparison)
        if (data.similarity_level === 0 && data.agent_updated) {
            console.log('[compare_agents] Similarity level 0 - showing confirmation page');
            console.log('[DEBUG] Attempting to show agentUpdated page');
            showPage('agentUpdated');
            console.log('[compare_agents] Switched to agent-updated page');
            console.log('[compare_agents] ===== END =====');
            return;
        }

        if (data.rawImageUpdated && data.rawImagePrev) {
            console.log('[compare_agents] Setting image sources at original size');

            // Create new images to get dimensions
            const img1 = new Image();
            const img2 = new Image();

            img1.onload = () => {
                console.log('[compare_agents] Previous agent image loaded:', img1.width, 'x', img1.height);
            };

            img2.onload = () => {
                console.log('[compare_agents] Updated agent image loaded:', img2.width, 'x', img2.height);
            };

            // Set sources - images will maintain original proportions
            img1.src = 'data:image/png;base64,' + data.rawImageUpdated;
            img2.src = 'data:image/png;base64,' + data.rawImagePrev;

            updatedAgentImage.src = img1.src;
            previousAgentImage.src = img2.src;


            // Show compare page after images are set
            showPage('compare');
            console.log('[compare_agents] Switched to compare page');
        } else {
            console.error('[compare_agents] Missing image data in response');
            alert('Failed to load comparison images. Please try again.');
        }

        console.log('[compare_agents] ===== END =====');
    });

    socket.on('game_finished', (data) => {
        console.log('[game_finished] ===== GAME COMPLETE =====');
        console.log('[game_finished] Total episodes:', data.total_episodes);
        console.log('[game_finished] Final agent index:', data.final_agent_index);
        
        // Update completion code with agent index
        const completionCodeElement = document.getElementById('completion-code');
        if (completionCodeElement && data.final_agent_index !== undefined) {
            completionCodeElement.textContent = `APPL${data.final_agent_index}`;
        }
        
        hideLoading();
        showPage('finish');
        console.log('[game_finished] Switched to finish page');
    });

    // Initialize
    console.log('[init] Initializing game...');
    console.log('[init] Player name:', playerName);
    console.log('[init] Canvas elements:', {
        agentVideo: !!canvases.agentVideo,
        overviewCanvas: !!canvases.overviewCanvas,
        previousVideo: !!canvases.previousVideo,
        updatedVideo: !!canvases.updatedVideo
    });

    // Hide backward button (not used in feedback interface)
    if (buttons.playBackward) {
        buttons.playBackward.style.display = 'none';
    }

    populateActionDropdown();
    showPage('welcome');
    console.log('[init] Initialization complete');
});