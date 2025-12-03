window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
                return {
                    color: '#0f172a',
                    weight: 1,
                    fillColor: feature.properties.fillColor || '#9ca3af',
                    fillOpacity: 0.8
                };
            }

            ,
        function1: function(feature, context) {
                return {
                    weight: 3,
                    color: '#e5e7eb',
                    fillOpacity: 0.95
                };
            }

            ,
        function2: function(feature, layer, context) {
            const kabkot = feature.properties.kabkot;
            const label = feature.properties.cluster_label;
            layer.bindTooltip(kabkot + ' – ' + label, {
                sticky: true
            });
        }

    }
});