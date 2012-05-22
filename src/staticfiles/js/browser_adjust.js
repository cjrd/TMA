function browser_adjust() {
    // make adjustments for webkit-based browsers
    if (jQuery.browser.webkit) {
        elems = document.getElementsByTagName("hr");
        elems[0].style.top="1.2em";
    }
}
