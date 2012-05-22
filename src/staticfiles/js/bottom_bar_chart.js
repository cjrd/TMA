function count_adjust(percent) {
	elems = document.getElementsByClassName("count");
        elems[0].width = percent.toString() + "%"
}

function hide_count_bar() {
	elems = document.getElementsByClassName("hidden");
	elems[0].style.display = 'none';
}

function show_count_bar() {
	elems = document.getElementsByClassName("hidden");
	if (elems[0].style.display == 'none') {
        	elems[0].style.display = 'block';
	}
}
