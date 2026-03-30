function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    
    if (isError) {
        toast.style.borderLeft = "4px solid #ef4444";
    } else {
        toast.style.borderLeft = "4px solid #10b981";
    }

    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

async function launchApp(type, btnElement) {
    btnElement.classList.add('loading');
    btnElement.disabled = true;

    try {
        const response = await fetch(`/api/launch/${type}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showToast("✅ " + result.message);
        } else {
            showToast("❌ Error: " + result.message, true);
        }
    } catch (error) {
        showToast("❌ Network Error: Could not connect to launcher.", true);
    } finally {
        setTimeout(() => {
            btnElement.classList.remove('loading');
            btnElement.disabled = false;
        }, 500);
    }
}
