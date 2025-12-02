window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
                const cl = feature.properties.cluster;
                const colors = {
                    0: '#1f77b4',
                    1: '#ff7f0e'
                };
                return {
                    color: '#ffffff',
                    weight: 1,
                    fillColor: colors[cl] || '#808080',
                    fillOpacity: 0.7
                };
            }

            ,
        function1: function(feature, context) {
                return {
                    weight: 3,
                    color: '#000000',
                    fillOpacity: 0.9
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