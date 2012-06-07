function showhelp(field) {
    var fieldid = '#ht-' + field;
    if (jQuery(fieldid).hasClass('selected')) {
        jQuery(fieldid).hide();
        jQuery(fieldid).removeClass('selected');
    }
    else {
        jQuery(fieldid).show();
        jQuery(fieldid).addClass('selected');
    }

}
jQuery(document).ready(function() {

    // TODO use this to display errors?
    jQuery("#xbtn").click(function() {
        jQuery("#notebar").hide();
    });

    jQuery('#id_upload_file').bind('change', function() {
        //this.files[0].size gets the size of your file.
        if (this.files[0].size > 20000000)
            alert("This file is too large -- maximum upload size is 20 Mb")
    });

    var form = jQuery("#initform");
    form.submit(function(e) {
        jQuery("#sendbutton").attr('disabled', true);
        jQuery("#wait_img").show();
        jQuery("#sb_div").append('<div class="help-text" style="text-align:right; width:auto;"> Note: this analysis may take several minutes</div>')
    });

    // show upload data screen initially
    jQuery("#data_toy").show();
    jQuery("#data_toy_button").addClass('selected');

    jQuery(".help-text").hide();


    var current_alg = jQuery("#id_std_algo").val();
    //        vertical expand and compression
    var vert_tabs = jQuery('.container');

    if (jQuery("#" + current_alg + "_adv_container").hasClass("notops")) {
        jQuery("#numtops").attr('disabled', true);
    }
    else {
        jQuery("#numtops").attr('disabled', false);
    }
    vert_tabs.hide(); // initially hidden?
    jQuery('#advanced_params_title').click(
        function() {
            if (jQuery(this).hasClass("active")) {
                jQuery(this).removeClass("active");
                jQuery(this).addClass("inactive")
            }
            else {
                jQuery(this).removeClass("inactive");
                jQuery(this).addClass("active")
            }

            var elem = "#" + current_alg + "_adv_container";
            if (jQuery(elem).hasClass('selected')) {
                jQuery(elem).hide();
                jQuery(elem).removeClass('selected');
            }
            else {
                jQuery(elem).show();
                jQuery(elem).addClass('selected')
            }
            return false;
        }
    );

    jQuery("#id_std_algo").change(function() {
        var old_elem = "#" + current_alg + "_adv_container";
        current_alg = jQuery("#id_std_algo").val()
        var next_elem = '#' + current_alg + "_adv_container";
        if (jQuery(old_elem).hasClass('selected')) {
            jQuery(old_elem).hide();
            jQuery(old_elem).removeClass('selected');
            jQuery(next_elem).addClass('selected');
            jQuery(next_elem).show();
        }

        if (jQuery(next_elem).hasClass("notops")) {
            jQuery("#numtops").attr('disabled', true);
        }
        else {
            jQuery("#numtops").attr('disabled', false);
        }

    });
});