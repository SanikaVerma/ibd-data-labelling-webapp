// IBD Patient Timeline Viewer — client-side rendering logic
// Data (ALL_EVENTS, FLARES) is injected by Python into window before this script runs.

var FIXED = [
    'imaging',
    'prescription',
    'physician_claim',
    'lab_test',
    'ambulatory_visit',
    'hospital_admission'
];

var LABELS = {
    'imaging':            'Diagnostic Imaging',
    'prescription':       'Prescription',
    'physician_claim':    'Physician Claim',
    'lab_test':           'Lab Test',
    'ambulatory_visit':   'Ambulatory Visit',
    'hospital_admission': 'Hospitalization'
};

var COLORS = {
    'imaging':            '#636EFA',
    'prescription':       '#EF553B',
    'physician_claim':    '#AB63FA',
    'lab_test':           '#19D3F3',
    'ambulatory_visit':   '#00CC96',
    'hospital_admission': '#FFA15A'
};

function filterEvents(filter) {
    if (filter === 'ibd')     return ALL_EVENTS.filter(function(e){ return  e.ibd_related; });
    if (filter === 'non_ibd') return ALL_EVENTS.filter(function(e){ return !e.ibd_related; });
    return ALL_EVENTS;
}

function buildHoverText(e) {
    var label = LABELS[e.event_type] || e.event_type || '';
    var t = (label ? '<b>event_type:</b> ' + label + '<br>' : '') +
            '<b>start:</b> ' + e.start_date.split('T')[0] +
            '<br><b>end:</b> ' + e.end_date.split('T')[0] + '<br>';
    try {
        var info = typeof e.event_info === 'string' ? JSON.parse(e.event_info) : e.event_info;
        Object.keys(info).forEach(function(k) {
            if (k === 'dummy') return;
            var v = String(info[k]);
            if (v.length > 40) {
                var chunks = [];
                for (var i = 0; i < v.length; i += 40) chunks.push(v.slice(i, i + 40));
                t += '<b>' + k + ':</b><br>' + chunks.join('<br>') + '<br>';
            } else {
                t += '<b>' + k + ':</b> ' + v + '<br>';
            }
        });
    } catch (err) {
        // Plain string event_info (e.g. "At: Hospital | Dx: ... | CMG: ...")
        var raw = e.event_info ? String(e.event_info).trim() : '';
        if (raw) {
            raw.split(' | ').forEach(function(part) {
                var ci = part.indexOf(': ');
                if (ci > 0) {
                    t += '<b>' + part.slice(0, ci) + ':</b> ' + part.slice(ci + 2) + '<br>';
                } else if (part) {
                    t += part + '<br>';
                }
            });
        }
    }
    if (e.source_dataset) t += '<b>source_dataset:</b> ' + e.source_dataset + '<br>';
    t += '<b>ibd_related:</b> ' + e.ibd_related;
    return t;
}

function processEvents(events) {
    var result = events.map(function(e) { return Object.assign({}, e); });

    result.forEach(function(e) {
        e._s = new Date(e.start_date).getTime();
        e._e = new Date(e.end_date).getTime();
        if (e._s === e._e) e._e += 86400000;
    });

    // Stagger same-day lab tests by 2 hours each so they don't overlap
    var labByDate = {};
    result.filter(function(e) { return e.event_type === 'lab_test'; })
          .forEach(function(e) {
              var k = e.start_date.split('T')[0];
              if (!labByDate[k]) labByDate[k] = [];
              labByDate[k].push(e);
          });
    Object.keys(labByDate).forEach(function(k) {
        var grp = labByDate[k];
        if (grp.length > 1) grp.forEach(function(e, i) {
            var offset = i * 7200000;
            e._s += offset;
            e._e += offset;
        });
    });

    return result;
}

// Move Plotly's modebar into the filter bar so it sits on the same row
function relocateModebar() {
    setTimeout(function() {
        var modebar   = document.querySelector('.js-plotly-plot .modebar-container');
        var filterBar = document.getElementById('filter-bar');
        if (modebar && filterBar && modebar.parentElement !== filterBar) {
            filterBar.appendChild(modebar);
        }
    }, 80);
}

var _chartReady = false;

function renderChart(filter) {
    var events = processEvents(filterEvents(filter));
    var ref    = ALL_EVENTS.length ? new Date(ALL_EVENTS[0].start_date).getTime() : Date.now();

    var traces = FIXED.map(function(et) {
        var te    = events.filter(function(e) { return e.event_type === et; });
        var dummy = te.length === 0;
        return {
            type:          'bar',
            orientation:   'h',
            name:          LABELS[et] || et,
            x:             dummy ? [0]   : te.map(function(e) { return e._e - e._s; }),
            base:          dummy ? [ref] : te.map(function(e) { return e._s; }),
            y:             dummy ? [et]  : te.map(function()  { return et; }),
            customdata:    dummy ? ['']  : te.map(function(e) { return buildHoverText(e); }),
            hovertemplate: dummy ? '<extra></extra>' : '%{customdata}<extra></extra>',
            textposition:  'none',
            marker:        { color: COLORS[et] || '#888', opacity: dummy ? 0 : 1 },
            width:         dummy ? 0 : 0.9,
            showlegend:    false
        };
    });

    var shapes = FLARES.map(function(f) {
        return {
            type: 'rect', xref: 'x', yref: 'paper',
            x0: f.start_date, x1: f.end_date,
            y0: 0, y1: 1,
            fillcolor: 'red', opacity: 0.3,
            layer: 'below', line: { width: 0 }
        };
    });

    var sMs  = ALL_EVENTS.map(function(e) { return new Date(e.start_date).getTime(); });
    var eMs  = ALL_EVENTS.map(function(e) { return new Date(e.end_date).getTime(); });
    var tMin = sMs.length ? Math.min.apply(null, sMs) - 864000000 : ref - 864000000;
    var tMax = eMs.length ? Math.max.apply(null, eMs) + 864000000 : ref + 864000000;

    var layout = {
        xaxis: { type: 'date', title: 'Time', range: [tMin, tMax] },
        yaxis: {
            title:         'Event Type',
            categoryorder: 'array',
            categoryarray: FIXED,
            tickvals:      FIXED,
            ticktext:      FIXED.map(function(et) { return LABELS[et] || et; })
        },
        barmode:       'overlay',
        showlegend:    false,
        height:        820,
        margin:        { l: 150, r: 30, t: 50, b: 60 },
        shapes:        shapes,
        hovermode:     'closest',
        bargap:        0.1,
        bargroupgap:   0.0,
        plot_bgcolor:  'white',
        paper_bgcolor: 'white'
    };

    var config = {
        responsive:             true,
        displayModeBar:         true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d']
    };

    if (!_chartReady) {
        Plotly.newPlot('plot', traces, layout, config);
        _chartReady = true;
        relocateModebar();
    } else {
        Plotly.react('plot', traces, layout, config);
    }
}

// Filter button handler — called by onclick in the HTML
window.setF = function(filter) {
    document.querySelectorAll('.fbtn').forEach(function(b) { b.className = 'fbtn off'; });
    var idx = { all: 0, ibd: 1, non_ibd: 2 };
    document.querySelectorAll('.fbtn')[idx[filter]].className = 'fbtn on';
    renderChart(filter);
};

// Load Plotly: use the copy inlined into this iframe's <head>. We deliberately
// do NOT borrow window.parent.Plotly — drawing into this iframe's div with the
// parent page's Plotly instance renders cross-document and comes out blank on
// the second load (once Gradio has lazy-loaded its own Plotly for other charts).
function initPlotly() {
    if (typeof Plotly !== 'undefined') {
        renderChart('all');
        return;
    }

    // CDN in <head> may still be loading — poll for up to 10 s
    var attempts = 0;
    var iv = setInterval(function() {
        if (typeof Plotly !== 'undefined') {
            clearInterval(iv);
            renderChart('all');
        } else if (++attempts > 100) {
            clearInterval(iv);
            document.getElementById('plot').innerHTML =
                '<p style="color:#ef4444;padding:20px;">Could not load chart library ' +
                '(no internet access). Contact your administrator.</p>';
        }
    }, 100);
}

initPlotly();
