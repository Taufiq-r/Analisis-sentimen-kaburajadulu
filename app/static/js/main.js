// main.js
// ---------------------------------------------
// Script utama untuk interaksi UI aplikasi web.
// ---------------------------------------------
// - Toggle sidebar (buka/tutup)
// - Responsif: sidebar otomatis tertutup di layar kecil
// - Menyesuaikan layout konten utama
// - Menyesuaikan posisi tombol toggle
// - Mengubah ikon tombol toggle
// ---------------------------------------------

$(document).ready(function() {
    const sidebar = $('#sidebar');
    const mainContent = $('#main-content');
    const toggleButton = $('#toggle-sidebar');
    const toggleIcon = toggleButton.find('i');

    // Fungsi untuk menyesuaikan posisi tombol toggle dan ikon
    function updateToggleButton() {
        var sidebarWidth = sidebar.width(); // Ambil lebar sidebar saat ini
        if (sidebar.hasClass('closed')) {
            toggleButton.css('left', '15px'); // Posisi saat sidebar tertutup
            toggleIcon.removeClass('fa-times').addClass('fa-bars');
        } else {
            // Posisi saat sidebar terbuka (lebar sidebar + sedikit spasi)
            toggleButton.css('left', (sidebarWidth + 15) + 'px'); 
            toggleIcon.removeClass('fa-bars').addClass('fa-times');
        }
    }

    // Fungsi untuk mengatur status sidebar berdasarkan lebar layar
    function handleResize() {
        const screenWidth = $(window).width();
        if (screenWidth > 768) {
            if (sidebar.hasClass('closed')) { // Hanya buka jika sebelumnya tertutup oleh resize
                sidebar.removeClass('closed');
                mainContent.removeClass('full-width');
            }
        } else {
            if (!sidebar.hasClass('closed')) { // Hanya tutup jika sebelumnya terbuka
                sidebar.addClass('closed');
                mainContent.addClass('full-width');
            }
        }
        updateToggleButton(); // Selalu update posisi tombol setelah resize
    }

    // Event listener untuk tombol toggle sidebar
    toggleButton.click(function() {
        sidebar.toggleClass('closed');
        mainContent.toggleClass('full-width');
        updateToggleButton(); // Update posisi dan ikon tombol
    });

    // Event listener untuk resize window
    $(window).resize(function() {
        handleResize();
    });

    // Atur kondisi awal saat halaman dimuat
    handleResize(); // Panggil handleResize untuk mengatur state awal dan posisi tombol

    // Bagian validasi form klasifikasi yang dikomentari bisa Anda aktifkan jika perlu
    // var klasifikasiForm = document.getElementById('klasifikasiForm');
    // if (klasifikasiForm) {
    //     klasifikasiForm.addEventListener('submit', function(e) {
    //         var ratio = document.getElementById('test_ratio').value;
    //         if (!ratio) {
    //             alert('Silakan pilih rasio data uji terlebih dahulu sebelum melakukan klasifikasi.');
    //             e.preventDefault();
    //         }
    //     });
    // }
});