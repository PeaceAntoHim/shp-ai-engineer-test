// Configuration
const API_BASE_URL = '/api/v1';

// State
let currentSection = 'upload';
let selectedAIMode = 'rule_based';
let aiCapabilities = {
    generative_available: false,
    rule_based_available: true,
    default_mode: 'rule_based'
};

// DOM Elements
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const uploadPrompt = document.getElementById('upload-prompt');
const uploadProgress = document.getElementById('upload-progress');
const uploadResult = document.getElementById('upload-result');
const imagePreview = document.getElementById('image-preview');
const previewImage = document.getElementById('preview-image');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    loadAICapabilities();
    showSection('upload');
});

// AI Mode Management
async function loadAICapabilities() {
    try {
        const response = await fetch(`${API_BASE_URL}/insights/ai-availability`);
        if (response.ok) {
            aiCapabilities = await response.json();
            setupAIModeUI();
        }
    } catch (error) {
        console.error('Failed to load AI capabilities:', error);
        setupAIModeUI(); // Use defaults
    }
}

function setupAIModeUI() {
    const ruleBasedBtn = document.getElementById('rule-based-btn');
    const generativeBtn = document.getElementById('generative-btn');
    const aiStatus = document.getElementById('ai-status');
    
    // Update button states
    if (aiCapabilities.generative_available) {
        generativeBtn.disabled = false;
        generativeBtn.classList.remove('text-gray-400', 'cursor-not-allowed');
        generativeBtn.classList.add('text-gray-700', 'hover:bg-gray-200');
        aiStatus.textContent = 'Gen AI Available';
        aiStatus.className = 'text-xs text-green-600';
    } else {
        generativeBtn.disabled = true;
        generativeBtn.classList.add('text-gray-400', 'cursor-not-allowed');
        generativeBtn.classList.remove('hover:bg-gray-200');
        aiStatus.textContent = 'Gen AI Unavailable';
        aiStatus.className = 'text-xs text-orange-600';
    }
    
    // Set default mode
    selectedAIMode = aiCapabilities.default_mode;
    selectAIMode(selectedAIMode, false);
}

function selectAIMode(mode, userTriggered = true) {
    const ruleBasedBtn = document.getElementById('rule-based-btn');
    const generativeBtn = document.getElementById('generative-btn');
    const description = document.getElementById('ai-mode-description');
    
    // Don't allow selecting generative mode if not available
    if (mode === 'generative' && !aiCapabilities.generative_available) {
        if (userTriggered) {
            alert('Generative AI is not available. Please configure OpenAI API key.');
        }
        return;
    }
    
    selectedAIMode = mode;
    
    // Reset all buttons
    [ruleBasedBtn, generativeBtn].forEach(btn => {
        btn.classList.remove('bg-blue-100', 'text-blue-700', 'border-blue-300');
        btn.classList.add('text-gray-700');
    });
    
    // Highlight selected button
    if (mode === 'rule_based') {
        ruleBasedBtn.classList.add('bg-blue-100', 'text-blue-700', 'border-blue-300');
        ruleBasedBtn.classList.remove('text-gray-700');
        description.textContent = 'Uses predefined patterns to analyze your spending data quickly and reliably.';
    } else if (mode === 'generative') {
        generativeBtn.classList.add('bg-blue-100', 'text-blue-700', 'border-blue-300');
        generativeBtn.classList.remove('text-gray-700');
        description.textContent = 'Uses advanced AI to provide creative and contextual insights about your spending patterns.';
    }
}

// Event Listeners
function setupEventListeners() {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    // Analytics period change
    document.getElementById('analytics-period').addEventListener('change', loadAnalytics);

    // Enter key for insight question
    document.getElementById('insight-question').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askInsight();
        }
    });

    // Keyboard shortcuts for closing modals
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeAllModals();
        }
    });

    // Click outside modal to close
    setupModalClickHandlers();
}

// Setup modal click handlers
function setupModalClickHandlers() {
    // Receipt detail modal
    document.getElementById('receipt-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeModal();
        }
    });

    // Image modal
    document.getElementById('image-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeImageModal();
        }
    });
}

// Enhanced modal closing functions
function closeAllModals() {
    closeModal();
    closeImageModal();
}

function closeModal() {
    const modal = document.getElementById('receipt-modal');
    modal.classList.add('hidden');
    
    // Clear modal content to free memory
    document.getElementById('modal-content').innerHTML = '';
}

function closeImageModal() {
    const modal = document.getElementById('image-modal');
    modal.classList.add('hidden');
    
    // Clear image source to free memory
    const modalImage = document.getElementById('modal-image');
    modalImage.src = '';
    modalImage.ondblclick = null; // Remove event listener
}

// Navigation
function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.add('hidden');
    });

    // Show selected section
    document.getElementById(`${sectionName}-section`).classList.remove('hidden');

    // Update navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('text-blue-600', 'bg-blue-50');
        btn.classList.add('text-gray-700');
    });

    // Find the button that was clicked and highlight it
    const clickedButton = document.querySelector(`button[onclick="showSection('${sectionName}')"]`);
    if (clickedButton) {
        clickedButton.classList.remove('text-gray-700');
        clickedButton.classList.add('text-blue-600', 'bg-blue-50');
    }

    currentSection = sectionName;

    // Close any open modals when switching sections
    closeAllModals();

    // Load section data
    if (sectionName === 'receipts') {
        loadReceipts();
    } else if (sectionName === 'insights') {
        loadAnalytics();
    }
}

// File Upload Functions
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('border-blue-500', 'bg-blue-50');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('border-blue-500', 'bg-blue-50');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('border-blue-500', 'bg-blue-50');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        imagePreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Upload file
    uploadFile(file);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Show progress
    uploadPrompt.classList.add('hidden');
    uploadProgress.classList.remove('hidden');
    uploadResult.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE_URL}/receipts/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showSuccess(data);
        } else {
            showError(data.detail || 'Upload failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        uploadPrompt.classList.remove('hidden');
        uploadProgress.classList.add('hidden');
    }
}

function showSuccess(data) {
    uploadResult.innerHTML = `
        <div class="bg-green-50 border border-green-200 rounded-md p-4">
            <div class="flex">
                <div class="flex-shrink-0">
                    <svg class="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                    </svg>
                </div>
                <div class="ml-3">
                    <h3 class="text-sm font-medium text-green-800">
                        Receipt processed successfully!
                    </h3>
                    <div class="mt-2 text-sm text-green-700">
                        <p><strong>Store:</strong> ${data.store_name}</p>
                        <p><strong>Total:</strong> $${data.total_amount.toFixed(2)}</p>
                        <p><strong>Items:</strong> ${data.items_count}</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    uploadResult.classList.remove('hidden');
}

function showError(message) {
    uploadResult.innerHTML = `
        <div class="bg-red-50 border border-red-200 rounded-md p-4">
            <div class="flex">
                <div class="flex-shrink-0">
                    <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
                    </svg>
                </div>
                <div class="ml-3">
                    <h3 class="text-sm font-medium text-red-800">Error</h3>
                    <p class="mt-2 text-sm text-red-700">${message}</p>
                </div>
            </div>
        </div>
    `;
    uploadResult.classList.remove('hidden');
}

// Receipts Functions
async function loadReceipts() {
    const receiptsList = document.getElementById('receipts-list');
    const loading = document.getElementById('receipts-loading');

    console.log('Loading receipts...');
    loading.classList.remove('hidden');
    receiptsList.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE_URL}/receipts/`);
        console.log('Response status:', response.status);
        
        const receipts = await response.json();
        console.log('Receipts data:', receipts);

        if (response.ok) {
            displayReceipts(receipts);
        } else {
            console.error('Failed to load receipts:', receipts);
            receiptsList.innerHTML = `
                <div class="p-4 text-center text-red-600">
                    Failed to load receipts: ${receipts.detail || 'Unknown error'}
                </div>
            `;
        }
    } catch (error) {
        console.error('Network error:', error);
        receiptsList.innerHTML = `
            <div class="p-4 text-center text-red-600">
                Network error: ${error.message}
            </div>
        `;
    } finally {
        loading.classList.add('hidden');
    }
}

function displayReceipts(receipts) {
    const receiptsList = document.getElementById('receipts-list');
    
    if (!receipts || receipts.length === 0) {
        receiptsList.innerHTML = `
            <div class="p-8 text-center text-gray-500">
                <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
                <p class="mt-2">No receipts uploaded yet</p>
            </div>
        `;
        return;
    }

    receiptsList.innerHTML = receipts.map(receipt => `
        <div class="p-6 hover:bg-gray-50">
            <div class="flex items-start space-x-4">
                <!-- Receipt Thumbnail -->
                <div class="flex-shrink-0">
                    <img 
                        src="${API_BASE_URL}/receipts/image/${receipt.id}" 
                        alt="Receipt thumbnail"
                        class="w-16 h-20 object-cover rounded border cursor-pointer hover:opacity-75 transition-opacity"
                        onclick="showReceiptImage(${receipt.id})"
                        onerror="handleImageError(this)"
                    />
                </div>
                
                <!-- Receipt Info -->
                <div class="flex-1 min-w-0">
                    <div class="flex justify-between items-start cursor-pointer" onclick="showReceiptDetail(${receipt.id})">
                        <div>
                            <h3 class="text-lg font-medium text-gray-900 truncate">${receipt.store_name || 'Unknown Store'}</h3>
                            <p class="text-sm text-gray-600">${new Date(receipt.receipt_date).toLocaleDateString()}</p>
                            <p class="text-sm text-gray-500 mt-1">${receipt.items ? receipt.items.length : 0} items</p>
                        </div>
                        <div class="text-right">
                            <p class="text-lg font-semibold text-gray-900">$${receipt.total_amount ? receipt.total_amount.toFixed(2) : '0.00'}</p>
                            ${receipt.tax_amount && receipt.tax_amount > 0 ? `<p class="text-sm text-gray-500">Tax: $${receipt.tax_amount.toFixed(2)}</p>` : ''}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

// Handle image loading errors
function handleImageError(img) {
    img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA2NCA4MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjY0IiBoZWlnaHQ9IjgwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0zMiA0MEM0My4wNDU3IDQwIDUyIDMxLjA0NTcgNTIgMjBDNTIgOC45NTQzIDQzLjA0NTcgMCAzMiAwQzIwLjk1NDMgMCAxMiA4Ljk1NDMgMTIgMjBDMTIgMzEuMDQ1NyAyMC45NTQzIDQwIDMyIDQwWiIgZmlsbD0iI0Q5RDBENyIvPgo8L3N2Zz4K';
}

function showReceiptImage(receiptId) {
    const modalImage = document.getElementById('modal-image');
    modalImage.src = `${API_BASE_URL}/receipts/image/${receiptId}`;
    
    // Add double-click to close
    modalImage.ondblclick = closeImageModal;
    
    document.getElementById('image-modal').classList.remove('hidden');
}

async function showReceiptDetail(receiptId) {
    try {
        const response = await fetch(`${API_BASE_URL}/receipts/${receiptId}`);
        const receipt = await response.json();

        if (response.ok) {
            displayReceiptModal(receipt);
        } else {
            alert('Failed to load receipt details');
        }
    } catch (error) {
        alert('Network error: ' + error.message);
    }
}

function displayReceiptModal(receipt) {
    const modalContent = document.getElementById('modal-content');

    modalContent.innerHTML = `
        <div class="space-y-4">
            <!-- Receipt Image Thumbnail -->
            <div class="text-center">
                <img 
                    src="${API_BASE_URL}/receipts/image/${receipt.id}" 
                    alt="Receipt image"
                    class="mx-auto max-w-48 max-h-64 object-contain border rounded cursor-pointer hover:opacity-75 transition-opacity"
                    onclick="showReceiptImage(${receipt.id})"
                    onerror="this.style.display='none'"
                />
                <p class="text-xs text-gray-500 mt-1">Click to view full size</p>
            </div>
            
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <h4 class="font-medium text-gray-900">Store</h4>
                    <p class="text-gray-600">${receipt.store_name}</p>
                    ${receipt.store_address ? `<p class="text-sm text-gray-500">${receipt.store_address}</p>` : ''}
                </div>
                <div>
                    <h4 class="font-medium text-gray-900">Date</h4>
                    <p class="text-gray-600">${new Date(receipt.receipt_date).toLocaleDateString()}</p>
                </div>
            </div>
            
            <div>
                <h4 class="font-medium text-gray-900 mb-2">Items</h4>
                <div class="border rounded-md divide-y max-h-64 overflow-y-auto">
                    ${receipt.items.length > 0 ? receipt.items.map(item => `
                        <div class="p-3 flex justify-between items-center">
                            <div>
                                <p class="font-medium">${item.item_name}</p>
                                <p class="text-sm text-gray-500">
                                    ${item.quantity} × $${item.unit_price.toFixed(2)}
                                    ${item.category ? ` • ${item.category}` : ''}
                                </p>
                            </div>
                            <p class="font-medium">$${item.total_price.toFixed(2)}</p>
                        </div>
                    `).join('') : `
                        <div class="p-3 text-center text-gray-500">
                            <p>No individual items detected</p>
                            <p class="text-xs mt-1">Only receipt total was captured</p>
                        </div>
                    `}
                </div>
            </div>
            
            <div class="border-t pt-4">
                <div class="flex justify-between items-center font-medium text-lg">
                    <span>Total</span>
                    <span>$${receipt.total_amount.toFixed(2)}</span>
                </div>
                ${receipt.tax_amount > 0 ? `
                    <div class="flex justify-between items-center text-sm text-gray-600">
                        <span>Tax</span>
                        <span>$${receipt.tax_amount.toFixed(2)}</span>
                    </div>
                ` : ''}
            </div>
        </div>
    `;

    document.getElementById('receipt-modal').classList.remove('hidden');
}

// Analytics and Insights functions (keeping existing ones)
async function loadAnalytics() {
    const period = document.getElementById('analytics-period').value;
    const resultDiv = document.getElementById('analytics-result');

    resultDiv.innerHTML = `
        <div class="animate-pulse">
            <div class="h-4 bg-gray-200 rounded mb-2"></div>
            <div class="h-4 bg-gray-200 rounded mb-2"></div>
            <div class="h-4 bg-gray-200 rounded"></div>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE_URL}/insights/analytics?days=${period}`);
        const analytics = await response.json();

        if (response.ok) {
            displayAnalytics(analytics);
        } else {
            resultDiv.innerHTML = `
                <div class="text-red-600 text-sm">
                    Failed to load analytics
                </div>
            `;
        }
    } catch (error) {
        resultDiv.innerHTML = `
            <div class="text-red-600 text-sm">
                Network error: ${error.message}
            </div>
        `;
    }
}

function displayAnalytics(analytics) {
    const resultDiv = document.getElementById('analytics-result');

    const totalSpent = analytics.total_spent || 0;
    const transactionCount = analytics.transaction_count || 0;
    const averageTransaction = analytics.average_transaction || 0;

    resultDiv.innerHTML = `
        <div class="space-y-4">
            <div class="bg-blue-50 p-4 rounded-lg">
                <h4 class="font-medium text-blue-900">Total Spending</h4>
                <p class="text-2xl font-bold text-blue-600">$${totalSpent.toFixed(2)}</p>
            </div>
            
            <div class="bg-green-50 p-4 rounded-lg">
                <h4 class="font-medium text-green-900">Number of Receipts</h4>
                <p class="text-2xl font-bold text-green-600">${transactionCount}</p>
            </div>
            
            <div class="bg-purple-50 p-4 rounded-lg">
                <h4 class="font-medium text-purple-900">Average per Receipt</h4>
                <p class="text-2xl font-bold text-purple-600">$${averageTransaction.toFixed(2)}</p>
            </div>
            
            ${analytics.top_categories && analytics.top_categories.length > 0 ? `
                <div>
                    <h4 class="font-medium text-gray-900 mb-2">Top Categories</h4>
                    <div class="space-y-2">
                        ${analytics.top_categories.map(cat => `
                            <div class="flex justify-between items-center bg-gray-50 p-2 rounded">
                                <span>${cat.category || 'Unknown'}</span>
                                <span class="font-medium">$${(cat.amount || 0).toFixed(2)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            
            ${analytics.daily_spending && analytics.daily_spending.length > 0 ? `
                <div>
                    <h4 class="font-medium text-gray-900 mb-2">Daily Spending</h4>
                    <div class="space-y-2 max-h-48 overflow-y-auto">
                        ${analytics.daily_spending.map(day => `
                            <div class="flex justify-between items-center bg-gray-50 p-2 rounded">
                                <span>${day.date || 'Unknown'}</span>
                                <span class="font-medium">$${(day.amount || 0).toFixed(2)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

async function askInsight() {
    const question = document.getElementById('insight-question').value.trim();
    if (!question) {
        alert('Please enter a question');
        return;
    }

    const resultDiv = document.getElementById('insight-result');

    // Show different loading messages based on AI mode
    const loadingMessage = selectedAIMode === 'generative' 
        ? 'Generative AI is thinking...' 
        : 'Rule-based AI is analyzing...';

    resultDiv.innerHTML = `
        <div class="bg-gray-50 p-4 rounded-lg">
            <div class="animate-pulse flex items-center">
                <div class="animate-spin-slow rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span class="ml-2 text-gray-600">${loadingMessage}</span>
            </div>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE_URL}/insights/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                question,
                ai_mode: selectedAIMode 
            })
        });

        const insight = await response.json();

        if (response.ok) {
            // Determine the color scheme based on AI mode used
            const isGenerative = insight.ai_mode_used === 'generative';
            const colorScheme = isGenerative 
                ? { bg: 'bg-purple-50', border: 'border-purple-400', text: 'text-purple-800', icon: 'text-purple-400', light: 'text-purple-700' }
                : { bg: 'bg-blue-50', border: 'border-blue-400', text: 'text-blue-800', icon: 'text-blue-400', light: 'text-blue-700' };

            const modeLabel = isGenerative ? 'Generative AI' : 'Rule-based AI';
            const confidenceBar = Math.round(insight.confidence * 100);

            resultDiv.innerHTML = `
                <div class="${colorScheme.bg} p-4 rounded-lg border-l-4 ${colorScheme.border}">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            <svg class="h-5 w-5 ${colorScheme.icon}" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" />
                            </svg>
                        </div>
                        <div class="ml-3 flex-1">
                            <div class="flex items-center justify-between mb-1">
                                <h4 class="text-sm font-medium ${colorScheme.text}">${modeLabel} Response</h4>
                                <div class="flex items-center text-xs ${colorScheme.light}">
                                    <span>Confidence: ${confidenceBar}%</span>
                                    <div class="ml-2 w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                                        <div class="h-full ${isGenerative ? 'bg-purple-500' : 'bg-blue-500'}" style="width: ${confidenceBar}%"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="mt-1 text-sm ${colorScheme.light}" style="white-space: pre-line">${insight.answer}</div>
                            ${insight.relevant_data && insight.relevant_data.length > 0 ? `
                                <div class="mt-2">
                                    <p class="text-xs ${colorScheme.light}">Based on ${insight.relevant_data.length} data point${insight.relevant_data.length !== 1 ? 's' : ''}</p>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
        } else {
            resultDiv.innerHTML = `
                <div class="bg-red-50 p-4 rounded-lg border-l-4 border-red-400">
                    <p class="text-sm text-red-700">${insight.detail || 'Failed to get insight'}</p>
                </div>
            `;
        }
    } catch (error) {
        resultDiv.innerHTML = `
            <div class="bg-red-50 p-4 rounded-lg border-l-4 border-red-400">
                <p class="text-sm text-red-700">Network error: ${error.message}</p>
            </div>
        `;
    }
}