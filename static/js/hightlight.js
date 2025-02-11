document.addEventListener('DOMContentLoaded', function() {
    const mseChart = Chart.getChart('mseChart');
    const psnrChart = Chart.getChart('psnrChart');
    const ssimChart = Chart.getChart('ssimChart');
    
    function highlightImage(index) {
        document.querySelectorAll('.cifar-image').forEach(img => {
            img.classList.remove('highlighted');
        });
        if (index !== null) {
            const targetImage = document.querySelector(`.cifar-image[data-index="${index}"]`);
            if (targetImage) targetImage.classList.add('highlighted');
        }
    }

    function highlightChartPoints(index) {
        [mseChart, psnrChart, ssimChart].forEach(chart => {
            if (!chart) return;
            
            chart.data.datasets[0].pointRadius = chart.data.datasets[0].data.map((_, i) => 
                i === index ? 6 : 3
            );
            chart.data.datasets[0].pointBorderWidth = chart.data.datasets[0].data.map((_, i) => 
                i === index ? 3 : 1
            );
            chart.update('none');
        });
    }

    // Add hover listeners to charts
    const addChartHoverListener = (chart) => {
        chart.canvas.addEventListener('mousemove', (event) => {
            const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, false);
            if (points.length) {
                const index = points[0].index;
                highlightImage(index);
                highlightChartPoints(index);
            } else {
                highlightImage(null);
                highlightChartPoints(null);
            }
        });

        chart.canvas.addEventListener('mouseleave', () => {
            highlightImage(null);
            highlightChartPoints(null);
        });
    };

    // Add hover listeners to images
    document.querySelectorAll('.cifar-image').forEach(img => {
        img.addEventListener('mouseenter', () => {
            const index = parseInt(img.dataset.index);
            highlightImage(index);
            highlightChartPoints(index);
        });

        img.addEventListener('mouseleave', () => {
            highlightImage(null);
            highlightChartPoints(null);
        });
    });

    // Initialize chart hover listeners
    [mseChart, psnrChart, ssimChart].forEach(chart => {
        if (chart) addChartHoverListener(chart);
    });
});