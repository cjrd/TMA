// horizontal tabs display
jQuery(document).ready(function() {
    var tabContainers = jQuery('div.tabs > div');
    tabContainers.hide();
    jQuery('div.tabs ul.tab_nav a').click(
        function () {
            if (jQuery(this).hasClass('selected')) {
                tabContainers.hide();
                jQuery('div.tabs ul.tab_nav a').removeClass('selected');
                return false;
            }
            tabContainers.hide().filter(this.hash).show();
            jQuery('div.tabs ul.tab_nav a').removeClass('selected');
            jQuery(this).addClass('selected');
            return false;
        });
});
