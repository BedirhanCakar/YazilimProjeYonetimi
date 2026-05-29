/**
 * Görüntü Sahteciliği Tespiti - Frontend Uygulaması
 * 
 * Bu script, dosya yükleme, görüntü önizleme ve analiz sonuçlarını
 * dinamik olarak göstermekten sorumludur.
 */

document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = uploadForm.querySelector('input[type=file]');
    const fileText = document.getElementById('file-text');
    const fileInfo = document.getElementById('file-info');
    const thresholdInput = document.getElementById('threshold');
    const thresholdValue = document.getElementById('threshold-value');

    // Threshold slider
    thresholdInput.addEventListener('input', (e) => {
        thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Dosya seçim
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            fileText.textContent = file.name;
            fileInfo.textContent = `✓ ${(file.size / 1024).toFixed(2)} KB seçildi`;
            fileInfo.classList.remove('hidden');

            // Önizleme göster
            const reader = new FileReader();
            reader.onload = (event) => {
                const previewImg = document.getElementById('preview-image');
                previewImg.src = event.target.result;
                document.getElementById('preview-section').classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    // Form gönderme
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!fileInput.files.length) {
            alert('Lütfen bir dosya seçin');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('threshold', thresholdInput.value);

        // Sonuç panelini göster ve loading başlat
        const resultsSection = document.getElementById('results-section');
        const loading = document.getElementById('loading');
        resultsSection.classList.remove('hidden');
        loading.classList.remove('hidden');

        // Tüm sonuç kartlarını gizle
        document.querySelectorAll('.result-card, .summary-card').forEach(card => {
            card.classList.add('hidden');
        });

        try {
            const response = await fetch('/upload/', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json().catch(() => null);
            if (!response.ok) {
                const message = data?.detail || `HTTP error! status: ${response.status}`;
                throw new Error(message);
            }

            loading.classList.add('hidden');

            // Sonuçları göster
            displayResults(data, parseFloat(thresholdInput.value));
        } catch (error) {
            loading.classList.add('hidden');
            alert('Hata: ' + error.message);
            console.error(error);
        }
    });

    /**
     * Analiz sonuçlarını UI'da göster
     */
    function displayResults(data, threshold) {
        // CNN Sonuçları
        if (data.ai_algorithms && data.ai_algorithms.cnn) {
            displayCNNResults(data.ai_algorithms.cnn, threshold);
        }

        // LSTM Sonuçları
        if (data.ai_algorithms && data.ai_algorithms.lstm) {
            displayLSTMResults(data.ai_algorithms.lstm, threshold);
        }

        // Klasik Algoritma Sonuçları
        if (data.classical_algorithms) {
            displayClassicalResults(data.classical_algorithms);
        }

        // Gelişmiş Analiz
        if (data.ai_algorithms) {
            displayAdvancedAnalysis(data.ai_algorithms);
        }

        // Özet
        displaySummary(data, threshold);
    }

    /**
     * CNN sonuçlarını göster
     */
    function displayCNNResults(cnnData, threshold) {
        const cnnResults = document.getElementById('cnn-results');
        const cnnBar = document.getElementById('cnn-bar');
        const cnnProb = document.getElementById('cnn-prob');
        const cnnVerdict = document.getElementById('cnn-verdict');

        const probability = cnnData.probability * 100;
        cnnBar.style.width = probability + '%';
        cnnProb.textContent = probability.toFixed(1) + '%';

        const isSuspicious = cnnData.is_suspicious || probability >= (threshold * 100);
        if (isSuspicious) {
            cnnVerdict.className = 'verdict suspicious';
            cnnVerdict.textContent = '⚠️ Sahte İçerik İhtimali Yüksek';
        } else {
            cnnVerdict.className = 'verdict safe';
            cnnVerdict.textContent = '✓ Sahte İçerik Belirtisi Düşük';
        }

        cnnResults.classList.remove('hidden');
    }

    /**
     * LSTM sonuçlarını göster
     */
    function displayLSTMResults(lstmData, threshold) {
        const lstmResults = document.getElementById('lstm-results');
        const lstmBar = document.getElementById('lstm-bar');
        const lstmProb = document.getElementById('lstm-prob');
        const lstmVerdict = document.getElementById('lstm-verdict');

        const probability = lstmData.probability * 100;
        lstmBar.style.width = probability + '%';
        lstmProb.textContent = probability.toFixed(1) + '%';

        const isSuspicious = lstmData.is_suspicious || probability >= (threshold * 100);
        if (isSuspicious) {
            lstmVerdict.className = 'verdict suspicious';
            lstmVerdict.textContent = '⚠️ Temporal Anomali Tespit Edildi';
        } else {
            lstmVerdict.className = 'verdict safe';
            lstmVerdict.textContent = '✓ Temporal Yapı Normal';
        }

        lstmResults.classList.remove('hidden');
    }

    /**
     * Klasik algoritma sonuçlarını göster ve copy-move çizgilerini çiz
     */
    function displayClassicalResults(classicalData) {
        const classicalResults = document.getElementById('classical-results');
        const algorithmsList = document.getElementById('algorithms-list');
        const canvas = document.getElementById('detection-canvas');
        const ctx = canvas.getContext('2d');
        const img = document.getElementById('preview-image');

        // Canvas'ı temizle
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Canvas boyutunu resmin orijinal boyutuna eşitle
        if (img.naturalWidth && img.naturalHeight) {
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
        }

        algorithmsList.innerHTML = '';

        const algorithms = ['sift', 'surf', 'akaze', 'orb'];
        // Algoritmalar için çizim renkleri
        const colors = {
            sift: 'rgba(239, 68, 68, 0.75)',  // Kırmızı
            surf: 'rgba(59, 130, 246, 0.75)',  // Mavi
            akaze: 'rgba(16, 185, 129, 0.75)', // Yeşil
            orb: 'rgba(245, 158, 11, 0.75)'    // Turuncu
        };

        algorithms.forEach(algo => {
            if (classicalData[algo] && classicalData[algo].available) {
                const data = classicalData[algo];
                const div = document.createElement('div');
                div.className = 'algo-item';

                let content = `<h4>${algo.toUpperCase()}</h4>`;
                content += `<p>Özellik Noktaları: ${data.keypoints_count || 0}</p>`;
                content += `<p>Eşleştirmeler: ${data.matches_count || 0}</p>`;

                if (data.region_detection) {
                    const regionData = data.region_detection;
                    const statusClass = regionData.detected ? 'suspicious' : 'safe';
                    const statusText = regionData.detected ? '✓ Şüpheli' : '✗ Normal';
                    content += `<p>Tespit: <span class="badge ${statusClass}">${statusText}</span></p>`;
                    content += `<p>Güven: ${(regionData.confidence * 100).toFixed(1)}%</p>`;
                    
                    // Copy-move koordinatları varsa canvas'a çiz
                    if (regionData.matches && regionData.matches.length > 0) {
                        ctx.strokeStyle = colors[algo];
                        ctx.lineWidth = Math.max(2, Math.round(canvas.width / 500)); // Çözünürlüğe göre kalınlık
                        ctx.fillStyle = colors[algo];

                        regionData.matches.forEach(match => {
                            const pt1 = match.pt1;
                            const pt2 = match.pt2;

                            // 1. Nokta
                            ctx.beginPath();
                            ctx.arc(pt1[0], pt1[1], ctx.lineWidth * 2, 0, 2 * Math.PI);
                            ctx.fill();

                            // 2. Nokta
                            ctx.beginPath();
                            ctx.arc(pt2[0], pt2[1], ctx.lineWidth * 2, 0, 2 * Math.PI);
                            ctx.fill();

                            // Çizgi
                            ctx.beginPath();
                            ctx.moveTo(pt1[0], pt1[1]);
                            ctx.lineTo(pt2[0], pt2[1]);
                            ctx.stroke();
                        });
                    }
                }

                div.innerHTML = content;
                algorithmsList.appendChild(div);
            }
        });

        classicalResults.classList.remove('hidden');
    }

    /**
     * Gelişmiş analiz (ELA, Frekans) sonuçlarını göster
     */
    function displayAdvancedAnalysis(aiData) {
        const advancedResults = document.getElementById('advanced-results');

        if (aiData.ela_analysis) {
            const elaScore = document.getElementById('ela-score');
            elaScore.textContent = (aiData.ela_analysis.score * 100).toFixed(1) + '%';
        }

        if (aiData.frequency_analysis) {
            const freqScore = document.getElementById('freq-score');
            freqScore.textContent = (aiData.frequency_analysis.score * 100).toFixed(1) + '%';
        }

        advancedResults.classList.remove('hidden');
    }

    /**
     * Genel özet göster
     */
    function displaySummary(data, threshold) {
        const summaryCard = document.getElementById('summary');
        const summaryContent = document.getElementById('summary-content');

        let avgScore = 0;
        let countScores = 0;

        // CNN skoru
        if (data.ai_algorithms && data.ai_algorithms.cnn) {
            avgScore += data.ai_algorithms.cnn.probability * 100;
            countScores++;
        }

        // LSTM skoru
        if (data.ai_algorithms && data.ai_algorithms.lstm) {
            avgScore += data.ai_algorithms.lstm.probability * 100;
            countScores++;
        }

        // ELA skoru
        if (data.ai_algorithms && data.ai_algorithms.ela_analysis) {
            avgScore += data.ai_algorithms.ela_analysis.score * 100;
            countScores++;
        }

        // Frekans skoru
        if (data.ai_algorithms && data.ai_algorithms.frequency_analysis) {
            avgScore += data.ai_algorithms.frequency_analysis.score * 100;
            countScores++;
        }

        avgScore = countScores > 0 ? avgScore / countScores : 0;

        // Klasik algoritma skoru
        let classicalScore = 0;
        let classicalConsensus = false;
        if (data.classical_algorithms && data.classical_algorithms.summary) {
            classicalScore = data.classical_algorithms.summary.overall_confidence * 100;
            classicalConsensus = data.classical_algorithms.summary.consensus_detected === true;
        }

        let html = `
            <div class="summary-stat">
                <span class="summary-stat-label">Ortalama Şüphe Puanı</span>
                <span class="summary-stat-value">${avgScore.toFixed(1)}%</span>
            </div>
            <div class="summary-stat">
                <span class="summary-stat-label">Klasik Algoritma Puanı</span>
                <span class="summary-stat-value">${classicalScore.toFixed(1)}%</span>
            </div>
        `;

        const aiDecision = data.ai_algorithms?.overall_suspicion?.final_decision === true;
        const isClassicalAlarm = classicalConsensus && classicalScore > 80;
        const isSuspicious = aiDecision || isClassicalAlarm;
        const verdictClass = isSuspicious ? 'suspicious' : 'safe';
        const verdictText = isSuspicious 
            ? '⚠️ Görüntü SAHTECİLİK BELİRTİLERİ İÇERİYOR'
            : '✓ Görüntü ORIJINAL GÖRÜNMEKTEDIR';

        html += `<div class="summary-verdict ${verdictClass}">${verdictText}</div>`;

        if (!isSuspicious && classicalConsensus) {
            html += `<div class="summary-note">Not: Klasik algoritma bir miktar şüphe tespit etti, ancak AI modelleri bunu desteklemiyor.</div>`;
        }

        if (aiDecision && !classicalConsensus) {
            html += `<div class="summary-note">AI modelleri sahtecilik sinyali verdi, ancak klasik yöntemlerde yeterli onay yok.</div>`;
        }

        summaryContent.innerHTML = html;
        summaryCard.classList.remove('hidden');
    }
});
