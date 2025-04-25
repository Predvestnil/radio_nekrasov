// static/js/app.js
document.addEventListener('DOMContentLoaded', function() {
    // Инициализация Telegram WebApp
    const tg = window.Telegram.WebApp;
    tg.enableClosingConfirmation(); // Подтверждение закрытия, если аудио проигрывается
    
    // DOM элементы для главного меню и загрузки файлов
    const mainMenuScreen = document.getElementById("main-menu-screen");
    const uploadArticleScreen = document.getElementById("upload-article-screen");
    const uploadFilesToggle = document.getElementById("upload-files-toggle");
    const sampleGalleryToggle = document.getElementById("sample-gallery-toggle");
    const uploadFilesSection = document.getElementById("upload-files-section");
    const sampleGallerySection = document.getElementById("sample-gallery-section");
    const articleFilesInput = document.getElementById("article-files");
    const filePreviewContainer = document.getElementById("file-preview-container");
    const uploadForm = document.getElementById("upload-form");
    const sampleGalleryContainer = document.getElementById("sample-gallery-container");
    
    // Массивы для хранения выбранных файлов и примеров
    let selectedFiles = [];
    let selectedSamples = [];
    
    // Обработчики переключения между разделами
    document.getElementById("listen-recordings-btn").onclick = () => {
        mainMenuScreen.classList.add("hidden");
        document.getElementById("years-menu").classList.remove("hidden");
        showYearsMenu();
    };
    
    document.getElementById("upload-article-btn").onclick = () => {
        mainMenuScreen.classList.add("hidden");
        uploadArticleScreen.classList.remove("hidden");
    };
    
    document.getElementById("back-to-main-from-upload").onclick = () => {
        uploadArticleScreen.classList.add("hidden");
        mainMenuScreen.classList.remove("hidden");
    };
    
    document.getElementById("back-to-main-from-years").onclick = () => {
        document.getElementById("years-menu").classList.add("hidden");
        mainMenuScreen.classList.remove("hidden");
    };
    
    document.getElementById("main-menu").onclick = () => {
        document.getElementById("audio-player").classList.add("hidden");
        mainMenuScreen.classList.remove("hidden");
    };
    
    // Переключение между загрузкой файлов и галереей примеров
    uploadFilesToggle.addEventListener("click", () => {
        uploadFilesToggle.classList.add("active");
        sampleGalleryToggle.classList.remove("active");
        uploadFilesSection.classList.remove("hidden");
        sampleGallerySection.classList.add("hidden");
    });
    
    sampleGalleryToggle.addEventListener("click", () => {
        sampleGalleryToggle.classList.add("active");
        uploadFilesToggle.classList.remove("active");
        sampleGallerySection.classList.remove("hidden");
        uploadFilesSection.classList.add("hidden");
        
        // Загружаем примеры, если еще не загружены
        if (sampleGalleryContainer.childElementCount === 0) {
            loadSampleGallery();
        }
    });
    
    // Глобальная переменная для хранения всех примеров
    let allSamples = [];
    let currentSlideIndex = 0;

    async function loadSampleGallery() {
        try {
            // Показываем индикатор загрузки
            sampleGalleryContainer.innerHTML = '<div class="loading" style="color: aquamarine;">Загрузка примеров...</div>';
            
            const response = await fetch('/api/samples');
            const data = await response.json();
            
            // Очищаем контейнер
            sampleGalleryContainer.innerHTML = '';
            
            // Проверяем, есть ли примеры
            if (!data.samples || data.samples.length === 0) {
                sampleGalleryContainer.innerHTML = '<div class="no-samples" style="color: aquamarine;">Примеры не найдены</div>';
                return;
            }
            
            // Сохраняем все примеры
            allSamples = data.samples;
            currentSlideIndex = 0;
            
            // Создаем слайды для каждого примера
            data.samples.forEach((sample, index) => {
                const slide = document.createElement('div');
                slide.className = 'sample-slide';
                slide.setAttribute('data-file', sample.name);
                slide.setAttribute('data-index', index);
                
                if (index === 0) {
                    slide.classList.add('active');
                }
                
                const img = document.createElement('img');
                img.src = sample.path; // Используем полное изображение вместо миниатюры
                img.alt = sample.name;
                slide.appendChild(img);
                
                const fileName = document.createElement('div');
                fileName.className = 'file-name';
                fileName.textContent = sample.name;
                slide.appendChild(fileName);
                
                // Добавляем индикатор выбора
                const selectionIndicator = document.createElement('div');
                selectionIndicator.className = 'selection-indicator';
                selectionIndicator.innerHTML = '✓';
                slide.appendChild(selectionIndicator);
                
                sampleGalleryContainer.appendChild(slide);
            });
            
            // Создаем точки для навигации
            createSliderDots(data.samples.length);
            
            // Добавляем обработчики для кнопок навигации
            document.getElementById('prev-slide').addEventListener('click', showPreviousSlide);
            document.getElementById('next-slide').addEventListener('click', showNextSlide);
            
            // Добавляем обработчик для кнопки "Выбрать этот пример"
            document.getElementById('select-sample').addEventListener('click', toggleCurrentSample);
            
            // Обновляем состояние выбора для первого слайда
            updateSelectButtonState();
            
        } catch (error) {
            console.error('Ошибка при загрузке примеров:', error);
            sampleGalleryContainer.innerHTML = '<div class="error" style="color: aquamarine;">Не удалось загрузить примеры. Попробуйте позже.</div>';
        }
    }

    function createSliderDots(count) {
        const dotsContainer = document.getElementById('slider-dots');
        dotsContainer.innerHTML = '';
        
        for (let i = 0; i < count; i++) {
            const dot = document.createElement('div');
            dot.className = 'slider-dot';
            if (i === 0) {
                dot.classList.add('active');
            }
            
            // Добавляем индикатор выбора для точки
            if (selectedSamples.includes(allSamples[i].name)) {
                dot.classList.add('selected');
            }
            
            dot.addEventListener('click', () => {
                showSlide(i);
            });
            
            dotsContainer.appendChild(dot);
        }
    }

    function showSlide(index) {
        // Получаем все слайды
        const slides = document.querySelectorAll('.sample-slide');
        const dots = document.querySelectorAll('.slider-dot');
        
        if (slides.length === 0) return;
        
        // Убираем активный класс со всех слайдов и точек
        slides.forEach(slide => slide.classList.remove('active'));
        dots.forEach(dot => dot.classList.remove('active'));
        
        // Устанавливаем новый индекс с учетом зацикливания
        currentSlideIndex = (index + slides.length) % slides.length;
        
        // Добавляем активный класс текущему слайду и точке
        slides[currentSlideIndex].classList.add('active');
        dots[currentSlideIndex].classList.add('active');
        
        // Обновляем состояние кнопки выбора
        updateSelectButtonState();
    }

    function showNextSlide() {
        showSlide(currentSlideIndex + 1);
    }

    function showPreviousSlide() {
        showSlide(currentSlideIndex - 1);
    }

    function toggleCurrentSample() {
        // Получаем текущий активный слайд
        const activeSlide = document.querySelector('.sample-slide.active');
        if (!activeSlide) return;
        
        const fileName = activeSlide.getAttribute('data-file');
        const slideIndex = parseInt(activeSlide.getAttribute('data-index'));
        
        // Проверяем, выбран ли уже этот пример
        if (selectedSamples.includes(fileName)) {
            // Если да, то убираем его из выбранных
            selectedSamples = selectedSamples.filter(name => name !== fileName);
            activeSlide.classList.remove('selected');
            
            // Обновляем соответствующую точку
            const dots = document.querySelectorAll('.slider-dot');
            if (dots[slideIndex]) {
                dots[slideIndex].classList.remove('selected');
            }
        } else {
            // Если нет, то добавляем его в выбранные
            selectedSamples.push(fileName);
            activeSlide.classList.add('selected');
            
            // Обновляем соответствующую точку
            const dots = document.querySelectorAll('.slider-dot');
            if (dots[slideIndex]) {
                dots[slideIndex].classList.add('selected');
            }
        }
        
        // Обновляем состояние кнопки
        updateSelectButtonState();
    }

    function updateSelectButtonState() {
        const activeSlide = document.querySelector('.sample-slide.active');
        if (!activeSlide) return;
        
        const fileName = activeSlide.getAttribute('data-file');
        const selectButton = document.getElementById('select-sample');
        
        if (selectedSamples.includes(fileName)) {
            selectButton.textContent = 'Отменить выбор';
            selectButton.classList.add('selected');
            activeSlide.classList.add('selected');
        } else {
            selectButton.textContent = 'Выбрать пример';
            selectButton.classList.remove('selected');
            activeSlide.classList.remove('selected');
        }
        
        // Обновляем счетчик выбранных примеров
        const selectedCounter = document.getElementById('selected-counter');
        if (selectedCounter) {
            selectedCounter.textContent = `Выбрано: ${selectedSamples.length}`;
        }
    }

    // Добавить вызов этой функции при изменении слайда
    document.addEventListener('DOMContentLoaded', function() {
        // Обновление состояния кнопки выбора при изменении слайда
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.attributeName === 'class' && 
                    mutation.target.classList.contains('sample-slide') && 
                    mutation.target.classList.contains('active')) {
                    updateSelectButtonState();
                }
            });
        });
        
        const config = { attributes: true, subtree: true };
        
        // Запускаем наблюдение за изменениями в контейнере слайдера
        setTimeout(() => {
            const container = document.getElementById('sample-gallery-container');
            if (container) {
                observer.observe(container, config);
            }
        }, 1000); // Даем время на инициализацию DOM
    });
    
    // Обработка загрузки файлов
    articleFilesInput.addEventListener("change", handleFileSelect);
    
    function handleFileSelect(event) {
        const files = event.target.files;
        if (!files.length) return;
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (!selectedFiles.some(f => f.name === file.name)) {
                selectedFiles.push(file);
            }
        }
        
        updateFilePreview();
    }
    
    function updateFilePreview() {
        filePreviewContainer.innerHTML = '';
        
        selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-preview-item';
            
            // Проверка типа файла для предпросмотра
            if (file.type.startsWith('image/')) {
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                fileItem.appendChild(img);
            } else {
                const icon = document.createElement('div');
                icon.className = 'file-icon';
                icon.textContent = getFileExtension(file.name).toUpperCase();
                fileItem.appendChild(icon);
            }
            
            const fileName = document.createElement('div');
            fileName.className = 'file-name';
            fileName.textContent = file.name;
            fileItem.appendChild(fileName);
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-file';
            removeBtn.innerHTML = '✕';
            removeBtn.addEventListener('click', () => {
                selectedFiles.splice(index, 1);
                updateFilePreview();
            });
            fileItem.appendChild(removeBtn);
            
            filePreviewContainer.appendChild(fileItem);
        });
    }
    
    function getFileExtension(filename) {
        return filename.split('.').pop();
    }
    
    // Обработка отправки формы
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const formData = new FormData();
        
        // Добавляем выбранные файлы
        selectedFiles.forEach(file => {
            formData.append("files", file);
        });
        
        // Добавляем выбранные примеры
        selectedSamples.forEach(sample => {
            formData.append("samples", sample);
        });
        
        // Добавляем выбранную модель
        const modelSelect = document.getElementById("model-select");
        formData.append("model", modelSelect.value);
        
        if (selectedFiles.length === 0 && selectedSamples.length === 0) {
            alert("Пожалуйста, выберите файлы или примеры для загрузки");
            return;
        }
        
        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });
            const result = await response.json();
            alert(`Результат: ${result.result}`);
        } catch (error) {
            console.error("Ошибка при отправке данных:", error);
            alert("Произошла ошибка при загрузке. Пожалуйста, попробуйте снова.");
        }
    });
    
    // DOM элементы для аудиоплеера и навигации
    const yearsMenu = document.getElementById('years-menu');
    const monthsMenu = document.getElementById('months-menu');
    const audioPlayer = document.getElementById('audio-player');
    const yearsButtons = document.getElementById('years-buttons');
    const monthsButtons = document.getElementById('months-buttons');
    const selectedYearText = document.getElementById('selected-year');
    const playerInfoText = document.getElementById('player-info');
    const audioElement = document.getElementById('audio-element');
    const playPauseButton = document.getElementById('play-pause');
    const rewindButton = document.getElementById('rewind');
    const forwardButton = document.getElementById('forward');
    const volumeSlider = document.getElementById('volume-slider');
    const volumeDownButton = document.getElementById('volume-down');
    const volumeUpButton = document.getElementById('volume-up');
    const backToYearsButton = document.getElementById('back-to-years');
    const backToMonthsButton = document.getElementById('back-to-months');
    const mainMenuButton = document.getElementById('main-menu');

    // Состояние приложения
    let currentState = {
        year: null,
        month: null,
        playing: false,
        volume: 1.0
    };

    // Функции для навигации между экранами
    function showYearsMenu() {
        yearsMenu.classList.remove('hidden');
        monthsMenu.classList.add('hidden');
        audioPlayer.classList.add('hidden');
        currentState.year = null;
        loadYears();
    }

    function showMonthsMenu(year) {
        yearsMenu.classList.add('hidden');
        monthsMenu.classList.remove('hidden');
        audioPlayer.classList.add('hidden');
        currentState.year = year;
        selectedYearText.textContent = `Выбранный год: ${year}`;
        loadMonths(year);
    }

    function showAudioPlayer(year, month, audioPath) {
        yearsMenu.classList.add('hidden');
        monthsMenu.classList.add('hidden');
        audioPlayer.classList.remove('hidden');
        currentState.year = year;
        currentState.month = month;
        playerInfoText.textContent = `${month} ${year}`;
        audioElement.src = audioPath;
        audioElement.volume = currentState.volume;
        updatePlayPauseButton();
    }

    // Загрузка данных с сервера
    async function loadYears() {
        try {
            const response = await fetch('/api/years');
            const data = await response.json();
            renderYearButtons(data.years);
        } catch (error) {
            console.error('Ошибка загрузки годов:', error);
            tg.showAlert('Не удалось загрузить список годов.');
        }
    }

    async function loadMonths(year) {
        try {
            const response = await fetch(`/api/months/${year}`);
            const data = await response.json();
            renderMonthButtons(data.months, year);
        } catch (error) {
            console.error('Ошибка загрузки месяцев:', error);
            tg.showAlert(`Не удалось загрузить список месяцев для ${year} года.`);
        }
    }

    async function loadAudio(year, month) {
        try {
            const response = await fetch(`/api/audio/${year}/${month}`);
            const data = await response.json();
            showAudioPlayer(year, month, data.path);
        } catch (error) {
            console.error('Ошибка загрузки аудио:', error);
            tg.showAlert(`Не удалось загрузить аудиозапись за ${month} ${year}.`);
        }
    }

    // Создание элементов интерфейса
    function renderYearButtons(years) {
        yearsButtons.innerHTML = '';
        years.forEach(year => {
            const button = document.createElement('button');
            button.textContent = year;
            button.className = 'glow-on-hover';
            button.addEventListener('click', () => showMonthsMenu(year));
            yearsButtons.appendChild(button);
        });
    }

    function renderMonthButtons(months, year) {
        monthsButtons.innerHTML = '';
        months.forEach(month => {
            const button = document.createElement('button');
            button.textContent = month;
            button.className = 'glow-on-hover';
            button.addEventListener('click', () => loadAudio(year, month));
            monthsButtons.appendChild(button);
        });
    }

    // Управление аудиоплеером
    function updatePlayPauseButton() {
        playPauseButton.textContent = audioElement.paused ? '▶️' : '⏸️';
    }

    playPauseButton.addEventListener('click', () => {
        if (audioElement.paused) {
            audioElement.play();
        } else {
            audioElement.pause();
        }
    });

    rewindButton.addEventListener('click', () => {
        audioElement.currentTime = Math.max(0, audioElement.currentTime - 15);
    });

    forwardButton.addEventListener('click', () => {
        audioElement.currentTime = Math.min(audioElement.duration, audioElement.currentTime + 15);
    });

    volumeSlider.addEventListener('input', () => {
        currentState.volume = parseFloat(volumeSlider.value);
        audioElement.volume = currentState.volume;
    });

    volumeDownButton.addEventListener('click', () => {
        currentState.volume = Math.max(0, currentState.volume - 0.1);
        volumeSlider.value = currentState.volume;
        audioElement.volume = currentState.volume;
    });

    volumeUpButton.addEventListener('click', () => {
        currentState.volume = Math.min(1, currentState.volume + 0.1);
        volumeSlider.value = currentState.volume;
        audioElement.volume = currentState.volume;
    });

    // События аудио-элемента
    audioElement.addEventListener('play', () => {
        currentState.playing = true;
        updatePlayPauseButton();
    });

    audioElement.addEventListener('pause', () => {
        currentState.playing = false;
        updatePlayPauseButton();
    });

    audioElement.addEventListener('ended', () => {
        currentState.playing = false;
        updatePlayPauseButton();
    });

    // Кнопки навигации
    backToYearsButton.addEventListener('click', showYearsMenu);
    mainMenuButton.addEventListener('click', () => {
        audioPlayer.classList.add('hidden');
        mainMenuScreen.classList.remove('hidden');
    });
    
    backToMonthsButton.addEventListener('click', () => {
        if (currentState.year) {
            showMonthsMenu(currentState.year);
        } else {
            showYearsMenu();
        }
    });
});