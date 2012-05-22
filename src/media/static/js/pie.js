function PieChart (elems) {
    var elements = elems;
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    var y = canvas.height / 2;
    var x = canvas.width / 2;
    var r = Math.min(canvas.width / 2, canvas.height / 2) * .9;
    var stringummie = "woohoo!";
    var total = 0;
    for (elem in elements) {
        total = total + elements[elem].value;
    }

    /*  public functions  */
    this.initialize = function() {
        hook_up_signals();
        init_canvas();
    };

    this.highlight = function(i) {
        var deg = 0;
        for (elem in elements.slice(0,i)) {
            deg = deg + (Math.PI*2 * elements[elem].value / total);
        }
        
        init_canvas();
        color_elem(deg, i);
        elements[i].highlight(); // usually css would take care of this, but in this case, onmouseover overrides it
    };

    this.unhighlight = function() {
        init_canvas();
    };
    
    /*  private functions  */ 
    function hook_up_signals() {
        canvas.addEventListener("click", mouse_click, false);
        canvas.addEventListener("mousemove", mouse_move, false);
        canvas.addEventListener("mouseout", init_canvas, false);
    }
    
    function init_canvas() {
        ctx.fillStyle = "rgb(209,209,209)";
        ctx.beginPath(); 
        ctx.arc(x, y, r, 0, Math.PI*2, true);
        ctx.fill();
        
        draw_pie_grid();

        for (elem in elements) {
            elements[elem].unhighlight();
        }
    }

    function draw_pie_grid() {
        ctx.strokeStyle = "white";
        ctx.lineWidth = "2";
        var deg = 0;
        for (elem in elements) {
            ctx.beginPath(); 
            ctx.arc(x, y, r, deg, deg + (Math.PI*2 * elements[elem].value / total), true);
            ctx.lineTo(x, y);
            deg = deg + (Math.PI*2 * elements[elem].value / total);
            ctx.stroke();
        }
    }

    function mouse_click(event) {
        var rv = find_elem(event);
        if (rv == null) {
            return false;
        }
        var event_elem = rv[1];
        window.location = elements[event_elem].link;
    }

    function mouse_move(event) {
        init_canvas();
        var rv = find_elem(event);
        if (rv == null) {
            document.body.style.cursor = "default";
            return false;
        }
        var event_degree = rv[0];
        var event_elem = rv[1];
        color_elem(event_degree, event_elem);
        
        // highlight element if present
        elements[event_elem].highlight();
        document.body.style.cursor = "pointer";
    }

    function find_elem (event) {
        var elem = canvas
        var offset_pos = { x: 0, y: 0}
        while (elem) {
            offset_pos.x += elem.offsetLeft;
            offset_pos.y += elem.offsetTop;
            elem = elem.offsetParent;
        }
        var ex = (event.clientX - offset_pos.x) - x; //(event.clientX - canvas.offsetLeft) - x;
        var ey = y - (document.body.scrollTop + event.clientY - offset_pos.y);//y - (event.clientY - canvas.offsetTop);

        var er = Math.sqrt(Math.pow(ex, 2) + Math.pow(ey, 2));
        
        if (er > r) {
            return null;
        }
        
        var e_deg = 0;
        if (ex == 0 && ey == 0) {
            e_deg = 0;
        } else if (ex >= 0 && ey >= 0) {
            e_deg = Math.asin(ey/er);
        } else if (ex >= 0 && ey <= 0) {
            e_deg = Math.asin(ey/er) + 2*Math.PI;
        } else if (ex < 0) { // ex < 0
            e_deg = -Math.asin(ey/er) + Math.PI;
        }
        
        e_deg = Math.PI*2 - e_deg;
        
        var deg = 0;
        for (elem in elements) {
            if (e_deg >= deg && e_deg <= (deg + (Math.PI*2 * elements[elem].value / total))) {
                return [deg, elem];
            }
            
            deg = deg + (Math.PI*2 * elements[elem].value / total);
        }
    }

    function color_elem(degree, elem_no) {
        ctx.fillStyle = "rgb(186,186,186)";
        ctx.beginPath();
        ctx.arc(x, y, r, degree, degree + (Math.PI*2 * elements[elem_no].value / total), false);
        ctx.lineTo(x, y);
        ctx.fill();
        draw_pie_grid();
    }
}

function generate_pie_elements(array) {
    var elements = new Array();
    
    var i = 0;
    for (i=0; i<array.length; i=i+1) {
        elements[i] = new PieElement(array[i][0], array[i][1], array[i][2]);
    }
    
    return elements;
}

function PieElement(v, l, i) {
    this.value = v;
    this.link = l;
    this.id = i;
    
    this.highlight = function() {
        var td = document.getElementById(this.id);
        if (td != null) {
            td.style.backgroundColor = '#BABABA';
        }
    }

    this.unhighlight = function() {
        var td = document.getElementById(this.id);
        if (td != null) {
            td.style.backgroundColor = '#D1D1D1';
        }
    }
}
