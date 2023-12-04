// this looks ugly but it works :rofl:  (also I must thank ChatGPT in helping me)

function convertTupleToIndex(tuple, K) {
    return tuple[0] * Math.pow(K, 3) + tuple[1] * Math.pow(K, 2) + tuple[2] * K + tuple[3];
}

function imageExists(url) {
    return fetch(url, { method: 'HEAD' })
        .then(res => {
            // Successful response (200-299) indicates the image exists
            return res.ok;
        })
        .catch(() => false); // Catch network errors or other issues
}

function addLeadingZeros(str, targetLength) {
    str = str.toString();
    while (str.length < targetLength) {
        str = '0' + str;
    }
    return str;
}

function round(num, K) {
    return Math.floor(num / K) * K;
}

document.addEventListener('DOMContentLoaded', function () {
    var birdData = [
        { text: 'Geococcyx', imgSrc: 'assets/birds/0.jpg' },
        { text: 'Pileated Woodpecker', imgSrc: 'assets/birds/1.jpg' },
        { text: 'Horned Lark', imgSrc: 'assets/birds/2.jpg' },
        { text: 'Blue Jay', imgSrc: 'assets/birds/3.jpg' },
        { text: 'Cardinal', imgSrc: 'assets/birds/4.jpg' }
    ];
    var dogData = [
        { text: 'Shetland Sheepdog', imgSrc: 'assets/dogs/0.jpg' },
        { text: 'Pomeranian', imgSrc: 'assets/dogs/1.jpg' },
        { text: 'Scotch Terrier', imgSrc: 'assets/dogs/2.jpg' },
        { text: 'Pug', imgSrc: 'assets/dogs/3.jpg' },
        { text: 'Papillon', imgSrc: 'assets/dogs/4.jpg' }
    ];

    var birdSeeds = [
        [235, 470, 705, 940],
        [235, 470, 705, 940],
        [470, 940, 1175, 1880],
        [235, 470, 1175, 1645],
        [235, 470, 1410, 1880],
        [235, 940, 1175, 1645],
        [235, 470, 555, 1665]
    ];
    var dogSeeds = [
        [235, 470, 705, 940]
    ];

    var dropdownData = [
        {
            id: 'subconcepts_head_bird',
            dispId: 'head_display',
            items: birdData,
        },
        {
            id: 'subconcepts_body_bird',
            dispId: 'body_display',
            items: birdData,
        },
        {
            id: 'subconcepts_wings_bird',
            dispId: 'wings_display',
            items: birdData,
        },
        {
            id: 'subconcepts_legs_bird',
            dispId: 'legs_display',
            items: birdData,
        },
        {
            id: 'subconcepts_prompts_bird',
            dispId: 'prompt_display',
            items: [
                { text: 'A photo of a [...] bird.'},
                { text: 'A photo of a [...] bird, knife palette oil painting style.'},
                { text: 'A photo of a cat with body of <b>[Wings]</b>.'},
                { text: 'A photo of a lion with body of <b>[Wings]</b>.'},
                { text: 'A cuddly and fluffy toy design, featuring a <b>[Head, Wings]</b> , perfect for a soft and adorable plushie.'},
                { text: 'A fantasy art style anatomy diagram showcasing the intricate internal and external structure of a mythical <b>[Head, Wings]</b> bird, with labels and details.'},
                { text: 'A robot designed in the shape of a <b>[Head, Wings]</b> bird, showcasing sleek and futuristic lines, with a blend of mechanical and avian features, perfect for a sci-fi themed illustration.'},
            ]
        }
        // Add more dropdown data objects here
    ];
    var dropdownDataDogs = [
        {
            id: 'subconcepts_upper_dog',
            dispId: 'upper_display_dog',
            items: dogData
        },
        {
            id: 'subconcepts_ears_dog',
            dispId: 'ears_display_dog',
            items: dogData
        },
        {
            id: 'subconcepts_lower_dog',
            dispId: 'lower_display_dog',
            items: dogData
        },
        {
            id: 'subconcepts_body_dog',
            dispId: 'body_display_dog',
            items: dogData
        },
        {
            id: 'subconcepts_prompts_dog',
            dispId: 'prompt_display_dog',
            items: [
                { text: 'A photo of a [...] dog.'},
//                { text: 'A robot designed in the shape of a [...] bird, showcasing sleek and futuristic lines, with a blend of mechanical and avian features, perfect for a sci-fi themed illustration.'},
            ]
        }
        // Add more dropdown data objects here
    ];

    var currentSelectedBirds = [0,0,0,0,0];
    var currentSelectedDogs = [0,0,0,0,0];

    var f = function (dropdown, dropdownIndex) {
        var ul = document.getElementById(dropdown.id);

        // on load
        [0,0,0,0,0].forEach(function (index) {
            var promptDisplay = document.getElementById(dropdown.dispId);
            promptDisplay.innerHTML = dropdown.items[index].text;
        });

        dropdown.items.forEach(function (item, index) {
            var li = document.createElement('li');
            var a = document.createElement('a');
            a.href = '#';

            if (item.imgSrc) {
                a.innerHTML = item.text + ' <img src="' + item.imgSrc + '" style="width: 64px; height: 64px; object-fit: cover;"  class="img-responsive" alt="' + item.text + '">';
            } else {
                a.innerHTML = item.text;
            }


            a.style.display = 'flex';
            a.style.alignItems = 'center';
            a.style.justifyContent = 'space-between';

            a.addEventListener('click', function (event) {
                event.preventDefault();
                // You can add any code here to handle the selection
                console.log(item.text + ' selected, index ' + index);

                var promptDisplay = document.getElementById(dropdown.dispId);
                promptDisplay.innerHTML = item.text;

                console.log(dropdownIndex);

                if (!dropdown.dispId.includes('dog')) {
                    currentSelectedBirds[dropdownIndex] = index;
                    // head, body, wings, legs
                    // 2, 0, 4, 6
                    var selectionIndices = [
                        currentSelectedBirds[1],
                        currentSelectedBirds[0],
                        currentSelectedBirds[2],
                        currentSelectedBirds[3]
                     ];

                    var promptIndex = currentSelectedBirds[4];
                    var folder = 'assets/crossgen/birds/' + promptIndex;
                    var seeds = birdSeeds[promptIndex];
                    var ignores = [];
                    if (promptIndex >= 4) {
                        ignores = [0, 3];
                    } else if (promptIndex >= 2) {
                        ignores = [0, 1, 3];
                    }
                    var imgPostfix = '';
                } else {
                    // 0: eye, 1: background, 2: neck, 3: ear, 4: body, 5:leg, 6: mouth/nose, 7: forehead
                    // [(0,7), (2,6), (4,5), 3]
                    // upper, ear, lower, body

                    currentSelectedDogs[dropdownIndex] = index;
                    var selectionIndices = [
                        currentSelectedDogs[0],
                        currentSelectedDogs[2],
                        currentSelectedDogs[3],
                        currentSelectedDogs[1]
                     ];
                    var promptIndex = currentSelectedDogs[4];
                    var folder = 'assets/crossgen/dogs/' + promptIndex;
                    var seeds = dogSeeds[promptIndex];
                    var ignores = [];
                    var imgPostfix = '_dog';
                }

                console.log(selectionIndices);

                [0,1,2,3].forEach(function (dispIndex) {
                    var imgElement = document.getElementById('disp' + (dispIndex + 1) + imgPostfix);
                    // console.log(imgElement);
                    // console.log(seeds);
                    // console.log(dispIndex);
                    var seed = seeds[dispIndex];
                    var selectionToIndex = convertTupleToIndex(selectionIndices, 5);
                    console.log('promptIndex', promptIndex);
                    console.log('ignores', ignores);
                    if (ignores.length != 0) { // if there are some index to ignore
                        for (let i = 0; i < ignores.length; i++) {
                            let v = ignores[i];
                            selectionIndices[v] = 0;
                        }
                        console.log('selection', selectionIndices);
                        selectionToIndex = convertTupleToIndex(selectionIndices, 5);
                        console.log(selectionToIndex, round(selectionToIndex, 5));
                        selectionToIndex = round(selectionToIndex, 5);
                    }
                    selectionToIndex = addLeadingZeros(selectionToIndex, 3);

                    var fn = folder + '/seed' + seed + '_' + selectionToIndex + '.jpg';
                    console.log(fn);

                    imageExists(fn)
                        .then(exists => {
                            if (exists) {
                                console.log('The new image exists. Updating src.');
                                imgElement.src = fn;
                            } else {
                                console.log('The new image does not exist. Keeping the original src.');
                            }
                        });
                });
            });

            li.appendChild(a);
            ul.appendChild(li);
        });
    }

    dropdownData.forEach(f);
    dropdownDataDogs.forEach(f);

    [1,2,3,4].forEach(function (item, dispIndex) {
        var imgElement = document.getElementById('disp' + item);
        var seed = birdSeeds[0][dispIndex];
        var selectionToIndex = addLeadingZeros(convertTupleToIndex([0,0,0,0], 5), 3);
        var folder = 'assets/crossgen/birds/0';
        var fn = folder + '/seed' + seed + '_' + selectionToIndex + '.jpg';
        console.log(fn);

        imageExists(fn)
            .then(exists => {
                if (exists) {
                    console.log('The new image exists. Updating src.');
                    imgElement.src = fn;
                } else {
                    console.log('The new image does not exist. Keeping the original src.');
                }
            });
    });

    [1,2,3,4].forEach(function (item, dispIndex) {
        var imgElement = document.getElementById('disp' + item + '_dog');
        var seed = dogSeeds[0][dispIndex];
        var selectionToIndex = addLeadingZeros(convertTupleToIndex([0,0,0,0], 5), 3);
        var folder = 'assets/crossgen/dogs/0';
        var fn = folder + '/seed' + seed + '_' + selectionToIndex + '.jpg';
        console.log(fn);

        imageExists(fn)
            .then(exists => {
                if (exists) {
                    console.log('The new image exists. Updating src.');
                    imgElement.src = fn;
                } else {
                    console.log('The new image does not exist. Keeping the original src.');
                }
            });
    });
});