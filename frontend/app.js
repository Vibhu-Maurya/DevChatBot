document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentMode = 'search'; // 'search' or 'ask'

    // DOM Elements
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    const sourceFilter = document.getElementById('sourceFilter');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const loader = document.getElementById('loader');
    
    // Result Elements
    const resultsArea = document.getElementById('resultsArea');
    const answerSection = document.getElementById('answerSection');
    const answerContent = document.getElementById('answerContent');
    const citationsSection = document.getElementById('citationsSection');
    const citationsGrid = document.getElementById('citationsGrid');
    const citationsTitle = document.getElementById('citationsTitle');
    const metadataContainer = document.getElementById('metadataContainer');
    const metaLatency = document.getElementById('metaLatency');
    const metaSources = document.getElementById('metaSources');
    const metaModelContainer = document.getElementById('metaModelContainer');
    const errorContainer = document.getElementById('errorContainer');

    // Tab Switching Logic
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            
            // Focus input naturally
            searchInput.focus();
        });
    });

    // Form Submission
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = searchInput.value.trim();
        if (!query) return;

        const source = sourceFilter.value;
        await performRequest(query, source, currentMode);
    });

    async function performRequest(query, source, mode) {
        // Reset UI
        answerSection.classList.add('hidden');
        citationsSection.classList.add('hidden');
        metadataContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
        resultsArea.classList.remove('hidden');
        loader.classList.remove('hidden');
        
        // Build payload
        const payload = { query };
        if (source) payload.source = source;
        if (mode === 'ask') payload.llm_provider = 'gemini';

        const endpoint = mode === 'search' ? '/search' : '/ask';
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `Server Error: ${response.status}`);
            }

            const data = await response.json();
            
            // Render Response
            if (mode === 'search') {
                renderSearchResults(data);
            } else {
                renderAskResults(data);
            }

        } catch (error) {
            errorContainer.textContent = error.message;
            errorContainer.classList.remove('hidden');
        } finally {
            loader.classList.add('hidden');
        }
    }

    function renderSearchResults(data) {
        // Metadata
        metaLatency.textContent = `Search Time: ${data.latency_ms.toFixed(1)}ms`;
        metaSources.textContent = `Found ${data.results.length} sources`;
        metaModelContainer.classList.add('hidden');
        metadataContainer.classList.remove('hidden');

        // Results Grid
        citationsGrid.innerHTML = '';
        citationsTitle.textContent = 'Search Results';
        
        data.results.forEach((item, index) => {
            const delay = index * 0.05;
            const card = document.createElement('a');
            card.href = item.url;
            card.target = '_blank';
            card.className = 'citation-card';
            card.style.animationDelay = `${delay}s`;
            
            card.innerHTML = `
                <div class="card-source">${item.source || 'Docs'}</div>
                <div class="card-title">${item.title}</div>
                <div class="card-snippet">${item.content.substring(0, 150)}...</div>
                <div class="card-domain">${item.domain}</div>
            `;
            citationsGrid.appendChild(card);
        });

        citationsSection.classList.remove('hidden');
    }

    function renderAskResults(data) {
        // Metadata
        metaLatency.textContent = `Total Time: ${data.total_ms.toFixed(1)}ms`;
        metaSources.textContent = `Retrieved ${data.sources.length} sources`;
        metaModelContainer.classList.remove('hidden');
        metadataContainer.classList.remove('hidden');

        // Render Markdown Answer
        if (typeof marked !== 'undefined') {
            answerContent.innerHTML = marked.parse(data.answer);
        } else {
            answerContent.textContent = data.answer;
        }
        answerSection.classList.remove('hidden');

        // Render Source Cards
        citationsGrid.innerHTML = '';
        citationsTitle.textContent = 'References';
        
        data.sources.forEach((item, index) => {
            const delay = index * 0.05 + 0.2; // slight delay after answer
            const card = document.createElement('a');
            card.href = item.url;
            card.target = '_blank';
            card.className = 'citation-card';
            card.style.animationDelay = `${delay}s`;
            
            // Extract domain and generic source from URL since sources array only has title/url
            let domain = '';
            try { domain = new URL(item.url).hostname; } catch(e){}
            
            card.innerHTML = `
                <div class="card-source">[${index + 1}] Citation</div>
                <div class="card-title">${item.title}</div>
                <div class="card-domain">${domain}</div>
            `;
            citationsGrid.appendChild(card);
        });

        if (data.sources.length > 0) {
            citationsSection.classList.remove('hidden');
        }
    }
});
