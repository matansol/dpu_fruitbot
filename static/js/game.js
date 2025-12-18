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
        playSequence: document.getElementById('btn-play-sequence'),
        pauseSequence: document.getElementById('btn-pause-sequence'),
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
    const loadingOverlay = document.getElementById('loading-overlay');
    const previousAgentImage = document.getElementById('previous-agent-image');
    const updatedAgentImage = document.getElementById('updated-agent-image');

    // --- GET PLAYER NAME FROM URL OR GENERATE RANDOM ---
    function getPlayerNameFromURL() {
        const urlParams = new URLSearchParams(window.location.search);
        const prolificId = urlParams.get('PROLIFIC_PID');
        
        if (prolificId) {
            console.log('Prolific ID:', prolificId);
            return prolificId;
        }
        
        // Generate random number between 1 and 100
        const randomId = Math.floor(Math.random() * 100) + 1;
        console.log('Generated random player ID:', randomId);
        return randomId.toString();
    }

    // --- STATE ---
    let currentPage = 'welcome';
    let episodeImages = [];
    let episodeActions = [];
    let episodePositions = [];  // Add positions array
    let currentActionIndex = 0;
    let userFeedback = [];
    let previousAgentImages = [];
    let updatedAgentImages = [];
    let totalScore = 0;
    let playerName = getPlayerNameFromURL();
    
    // Playback state
    let isPlaying = false;
    let playbackInterval = null;

    const ACTION_NAMES = {
        0: "LEFT ←",
        1: "UP ↑",
        2: "RIGHT →",
        // 3: "THROW",
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
        console.log('[drawImageOnCanvas] Called with:', {
            canvasId: canvas?.id,
            canvasExists: !!canvas,
            imageDataLength: base64Image?.length,
            imagePrefix: base64Image?.substring(0, 50)
        });
        
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
            console.log('[drawImageOnCanvas] Image loaded successfully:', {
                canvasId: canvas.id,
                imageWidth: img.width,
                imageHeight: img.height,
                canvasWidth: canvas.width,
                canvasHeight: canvas.height
            });
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            console.log('[drawImageOnCanvas] Image drawn to canvas');
        };
        
        img.onerror = (error) => {
            console.error('[drawImageOnCanvas] Image failed to load:', error);
            console.error('[drawImageOnCanvas] Image src length:', img.src?.length);
            console.error('[drawImageOnCanvas] Image src prefix:', img.src?.substring(0, 100));
        };
        
        img.src = 'data:image/jpeg;base64,' + base64Image;
        console.log('[drawImageOnCanvas] Image src set, loading...');
    }

    function playVideoSequence(canvas, images, fps = 10, onComplete) {
        console.log('[playVideoSequence] Starting playback:', {
            canvasId: canvas?.id,
            numImages: images?.length,
            fps: fps
        });
        
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
                console.log('[playVideoSequence] Playback complete');
                if (onComplete) onComplete();
                return;
            }

            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                console.log(`[playVideoSequence] Frame ${frameIndex + 1}/${images.length} drawn`);
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
        switch(action) {
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
                
                // Position symbol at bottom 1/10 of canvas
                const symbolY = canvasHeight - (canvasHeight / 10);
                
                // Get x position from episode data, or use center as fallback
                const symbolX = episodePositions[index] !== undefined 
                    ? episodePositions[index] 
                    : canvases.overviewCanvas.width / 2;
                
                drawActionSymbol(ctx, displayAction, symbolX, symbolY, 40);
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
        console.log('[startGame] Button clicked');
        console.log('[startGame] Player name:', playerName);
        console.log('[startGame] Emitting start_game event');
        socket.emit('start_game', { playerName: playerName });
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
        
        isPlaying = true;
        buttons.playSequence.style.display = 'none';
        buttons.pauseSequence.style.display = 'flex';
        
        playbackInterval = setInterval(() => {
            if (currentActionIndex < episodeActions.length - 1) {
                showOverviewAction(currentActionIndex + 1);
            } else {
                // Reached end, reset to beginning and continue playing
                showOverviewAction(0);
            }
        }, 100); // 0.1 seconds per frame
    });

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
            // At beginning, cycle to end
            showOverviewAction(episodeActions.length - 1);
        }
    });

    buttons.nextAction.addEventListener('click', () => {
        stopPlayback(); // Stop auto-play when manually navigating
        if (currentActionIndex < episodeActions.length - 1) {
            showOverviewAction(currentActionIndex + 1);
        } else {
            // At end, cycle to beginning
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
        socket.emit('next_episode', { playerName: playerName });
        showPage('agentPlay');
        resetAgentPlayPage();
    });

    function resetAgentPlayPage() {
        episodeImages = [];
        episodeActions = [];
        episodePositions = [];  // Reset positions
        currentActionIndex = 0;
        userFeedback = [];
        totalScore = 0;
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
        if (data.positions) episodePositions.push(...data.positions);
        if (data.score !== undefined) totalScore = data.score;
        
        if (totalScoreElement) {
            totalScoreElement.textContent = totalScore.toFixed(1);
        }

        // If this is the final batch, transition to playback
        if (data.is_final && currentPage === 'agentPlay' && episodeImages.length > 0) {
            console.log('[episode_batch] Final batch received, starting playback with', episodeImages.length, 'frames');
            playVideoSequence(canvases.agentVideo, episodeImages, 10, () => {
                console.log('[episode_batch] Episode playback complete');
                buttons.playVideo.textContent = 'Episode Complete';
                setTimeout(() => {
                    showPage('overview');
                    populateActionDropdown();
                    showOverviewAction(0);
                }, 1500);
            });
        } else {
            console.log('[episode_batch] Batch accumulated, waiting for more...');
        }
        
        console.log('[episode_batch] ===== END =====');
    });

    socket.on('episode_data', (data) => {
        console.log('[episode_data] ===== RECEIVED =====');
        console.log('[episode_data] Received episode data:', data);
        console.log('[episode_data] Images array length:', data.images?.length);
        console.log('[episode_data] Actions array length:', data.actions?.length);
        console.log('[episode_data] Positions array length:', data.positions?.length);
        
        // Only use episode_data if we haven't received batches
        // (backwards compatibility or fallback)
        if (episodeImages.length === 0) {
            episodeImages = data.images || [];
            episodeActions = data.actions || [];
            episodePositions = data.positions || [];
            totalScore = data.score || 0;
            
            if (totalScoreElement) {
                totalScoreElement.textContent = totalScore.toFixed(1);
            }

            if (currentPage === 'agentPlay' && episodeImages.length > 0) {
                console.log('[episode_data] Playing video sequence with', episodeImages.length, 'frames');
                playVideoSequence(canvases.agentVideo, episodeImages, 10, () => {
                    console.log('[episode_data] Episode playback complete');
                    buttons.playVideo.textContent = 'Episode Complete';
                    setTimeout(() => {
                        showPage('overview');
                        populateActionDropdown();
                        showOverviewAction(0);
                    }, 1500);
                });
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
        console.log('[compare_agents] Has rawImage1:', !!data.rawImage1);
        console.log('[compare_agents] Has rawImage2:', !!data.rawImage2);
        
        // Hide loading screen
        hideLoading();
        
        if (data.rawImage1 && data.rawImage2) {
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
            img1.src = 'data:image/png;base64,' + data.rawImage1;
            img2.src = 'data:image/png;base64,' + data.rawImage2;
            
            previousAgentImage.src = img1.src;
            updatedAgentImage.src = img2.src;
            
            // Show compare page after images are set
            showPage('compare');
            console.log('[compare_agents] Switched to compare page');
        } else {
            console.error('[compare_agents] Missing image data in response');
            alert('Failed to load comparison images. Please try again.');
        }
        
        console.log('[compare_agents] ===== END =====');
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
    
    populateActionDropdown();
    showPage('welcome');
    console.log('[init] Initialization complete');
});