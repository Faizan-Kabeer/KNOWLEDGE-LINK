import { API, apiFetch } from '../core/api.js';
import { dom } from '../core/dom.js';
import { populateModelSelect } from './models.js';

export function initTrainingListeners() {
  dom.trainStartBtn.addEventListener('click', async () => {
    const modelName = dom.trainNameInput.value.trim();
    const file = dom.trainFileInput.files[0];
    
    if (!modelName || !file) {
      alert("Please provide a model name and select a dataset file.");
      return;
    }
    
    dom.trainStartBtn.disabled = true;
    dom.trainProgArea.classList.remove('hidden');
    dom.trainLog.textContent = "Uploading dataset file...\n";
    dom.trainProgBar.style.width = '0%';
    dom.trainStatusTxt.textContent = "Uploading...";
    dom.trainEpochTxt.textContent = "";
    dom.metricLoss.textContent = "--";

    // 1. Upload file
    try {
      const formData = new FormData();
      formData.append('model_name', modelName);
      formData.append('file', file);
      
      const mappingFile = dom.trainMappingInput.files[0];
      if (mappingFile) {
        formData.append('mapping_file', mappingFile);
      }

      const uploadRes = await fetch(API + '/training/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!uploadRes.ok) {
        const err = await uploadRes.json();
        throw new Error(err.detail || 'Upload failed');
      }
      
      dom.trainLog.textContent += "Upload successful. Starting training session...\n";
    } catch (err) {
      dom.trainLog.textContent += `[ERROR] ${err.message}\n`;
      dom.trainStatusTxt.textContent = "Upload Failed";
      dom.trainStatusTxt.style.color = "red";
      dom.trainStartBtn.disabled = false;
      return;
    }

    // 2. Start SSE Stream
    dom.trainStatusTxt.textContent = "Connecting to Modal GPU...";
    const urlParams = new URLSearchParams({ model_name: modelName });
    const eventSource = new EventSource(API + `/training/stream?${urlParams.toString()}`);
    
    eventSource.onmessage = function(event) {
      const data = JSON.parse(event.data);
      
      if (data.error) {
        dom.trainLog.textContent += `[ERROR] ${data.error}\n`;
        dom.trainStatusTxt.textContent = "Failed";
        dom.trainStatusTxt.style.color = "red";
        dom.trainStartBtn.disabled = false;
        eventSource.close();
        return;
      }
      
      if (data.status) {
        dom.trainLog.textContent += `[*] ${data.status}\n`;
        if (data.status === "done") {
          dom.trainStatusTxt.textContent = "Training Complete!";
          dom.trainStatusTxt.style.color = "var(--neon)";
          dom.trainStartBtn.disabled = false;
          
          // Refresh models list
          apiFetch('/models').then(res => {
            populateModelSelect(res.models, res.active_model);
          });
          
          eventSource.close();
        } else {
          dom.trainStatusTxt.textContent = data.status;
        }
      }
      
      if (data.epoch) {
        dom.trainEpochTxt.textContent = `Epoch ${data.epoch} / ${data.total_epochs}`;
        dom.trainProgBar.style.width = `${(data.epoch / data.total_epochs) * 100}%`;
        
        dom.metricLoss.textContent = data.loss;
        dom.trainLog.textContent += `[Epoch ${data.epoch}] Loss: ${data.loss}\n`;
      }
      
      // Auto scroll log to bottom
      dom.trainLog.scrollTop = dom.trainLog.scrollHeight;
    };
    
    eventSource.onerror = function(err) {
      console.error("SSE Error:", err);
      dom.trainLog.textContent += `[ERROR] Connection lost or training failed.\n`;
      dom.trainStatusTxt.textContent = "Connection Error";
      dom.trainStatusTxt.style.color = "red";
      dom.trainStartBtn.disabled = false;
      eventSource.close();
    };
  });
}
