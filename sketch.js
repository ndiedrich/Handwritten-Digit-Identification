var currPre = -1;

var sketch = function(p) {

p.pixelData = [];
p.PIC_DIM = 32;

p.saved = [];
p.saveLabels = [];
p.shouldShow = 0;
p.numSaved = 0;
p.nn = {};

p.setup = function() {

    p.drawHere = p.createCanvas(400, 400);

    for(var i=0; i<p.PIC_DIM; i++) {

        p.toAdd = [];

        for(var j=0; j<p.PIC_DIM; j++) {

            p.toAdd[j] = 0;

        }

        p.pixelData.push(p.toAdd);
    

    }

    p.button = p.createButton('Submit Number');
    p.button.position(10, 550);
    p.button.mousePressed(p.submitPressed);

    p.button = p.createButton('Load Standard Dataset');
    p.button.position(150, 550);
    p.button.mousePressed(p.doLoad);

    p.button2 = p.createButton('Train on Submissions');
    p.button2.position(10, 600);
    p.button2.mousePressed(p.submitTrain);

    p.button3 = p.createButton('Predict Current Digit');
    p.button3.position(10, 650);
    p.button3.mousePressed(p.submitPredict);

    p.button4 = p.createButton('Download Model Config');
    p.button4.position(10, 700);
    p.button4.mousePressed(p.doSave);

    p.in = p.createInput('Type # Here');
    p.in.position(10, 520);

}

p.doSave = async function() {

    await p.nn.save('downloads://digit-NN');
    
}

p.doLoad = async function() {

    // This is currently the best data set I have available to start
    p.nn = await tf.loadLayersModel('resources/digit-NN.json');
    console.log('Load complete');

}

p.draw = function() {

    p.stroke(255);

    p.background(255);

    for(var i=0; i<p.PIC_DIM; i++) {

        for(var j=0; j<p.PIC_DIM; j++) {

            if(p.pixelData[i][j] == 1) {
                p.fill(0);
            } else {
                p.fill(230);
            }
            p.rect(i*p.width/p.PIC_DIM, j*p.height/p.PIC_DIM, p.width/p.PIC_DIM-1, p.height/p.PIC_DIM-1);

        }

    }

    if(p.shouldShow != 0) {

        for(var save = 0; save < p.saved.length; save++) {

            for(var i=0; i<p.PIC_DIM; i++) {

                for(var j=0; j<p.PIC_DIM; j++) {

                    // show saved

                }
            }

        }

    }

}

p.mouseDragged = function() {


    if(p.mouseX >= 0 && p.mouseX <= p.width &&
        p.mouseY >= 0 && p.mouseY <= p.height) {

    p.xPos = p.floor((p.mouseX / p.width) * p.PIC_DIM);
    p.yPos = p.floor((p.mouseY / p.height) * p.PIC_DIM);

    p.pixelData[p.xPos][p.yPos] = 1;

    if(p.xPos > 0){
        p.pixelData[p.xPos-1][p.yPos] = 1;
    }

    if(p.yPos > 0){
        p.pixelData[p.xPos][p.yPos-1] = 1;
    }

    if(p.xPos < p.PIC_DIM){
        p.pixelData[p.xPos+1][p.yPos] = 1;
    }

    if(p.yPos < p.PIC_DIM){
        p.pixelData[p.xPos][p.yPos+1] = 1;
    }

    } else return false;

}

p.submitPressed = function() {

    p.saved.push(p.pixelData);

    // reset pixel data

    p.pixelData = [];

    for(var i=0; i<p.PIC_DIM; i++) {

        p.toAdd = [];

        for(var j=0; j<p.PIC_DIM; j++) {

            p.toAdd[j] = 0;

        }

        p.pixelData.push(p.toAdd);

    }

    p.numSaved++;

    console.log(" --- Sample " + p.saved.length + " Saved --- ");

    p.shouldShow = 1;

    p.saveLabels.push(p.in.value());
    p.in.clear;


}


p.submitTrain = function() {

    // process savelabels into proper format. 
    var len = p.saveLabels.length;

    var res = [];

    for(var i=0; i<len; i++) {

        var round = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001];
        round[p.saveLabels[i]] = .991;

        for(var j=0; j<round.length; j++) {
            res.push(round[j]);
        }

    }

    console.log(p.saved, p.numsaved, res);

    p.nn = brain(p.saved, p.numSaved, res);

}

p.submitPredict = function() {

    var toPredict = [];
    toPredict.push(p.pixelData);
    var t = tf.tensor(toPredict);
    const use = tf.reshape(t, [1, 1024]);

    var results = p.nn.predict(use).dataSync();

    var max = 0;
    var maxToShow = -1;

    for(var i=0; i<results.length; i++) {
    
        if(results[i] > max) {
            max = results[i];
            maxToShow = i;
        }

        console.log(results[i]);

    }

    currPre = maxToShow;

    // reset pixel data

    p.pixelData = [];

    for(var i=0; i<p.PIC_DIM; i++) {

        p.toAdd = [];

        for(var j=0; j<p.PIC_DIM; j++) {

            p.toAdd[j] = 0;

        }

        p.pixelData.push(p.toAdd);

    }

}

}

let writingTest = new p5(sketch);

var atAcc = 0;

    var drawProgressBar = function(d) {

        d.font = {};
        d.finished = 0;

        d.preload = function() {
            d.font = d.loadFont('resources/Arial.ttf');
        }

        d.setup = function() {

            d.newC = d.createCanvas(160, 40);
            d.newC.position(160, 600);
            d.textFont(d.font);
            d.textSize(12);
            d.textAlign(d.center, d.center);

        }

        d.draw = function() {
    
            d.clear();

            
            if(d.finished == 1) {

                d.stroke(0);
                d.fill(0, atAcc*255, 0);
                d.rect(0, 0, 150 * (atAcc), 30);

                d.fill(0);
                d.text('Training Complete', 20, 20);

            } else {

                d.stroke(0);

                d.fill(255)
                d.rect(0, 0, 150, 30);

                d.fill(255 - (atAcc*255), atAcc*150, 0);
                d.rect(0, 0, 150 * (atAcc), 30);

            }

        }

        d.finish = function() {
            d.finished = 1;
        }
    }


var progress = new p5(drawProgressBar);

var showPrediction = function(pr) {

    pr.font = {};
    pr.prediction;

    pr.preload = function() {
        pr.font = pr.loadFont('Arial.ttf');
    }

    pr.setup = function() {
        pr.resultC = pr.createCanvas(210, 40);
        pr.resultC.position(160, 650);
        pr.textFont(pr.font);
        pr.textSize(14);
        pr.textAlign(pr.center, pr.center);
    }

    pr.draw = function() {

        pr.clear();

        pr.fill(255)
        pr.rect(0, 0, 200, 30);

        

        if(currPre != -1) {

            pr.stroke(0);

            pr.fill(0);
            pr.text('Digit Indetified: ' + currPre, 10, 20);

        } 
    }

}

var PredictionArea = new p5(showPrediction);

var brain = function(inputData, numInput, inputLabels) {

    // correctly shaping data 
    var d = tf.tensor(inputData);
    const data = tf.reshape(d, [numInput, 1024]);
    const labels = tf.reshape(inputLabels, [numInput, 10]);
    
    console.log(tf.version);
    
    // intialize input
    //const inp = tf.input({shape: [1024]});
    
    // initialize 2 hidden layers
    const h1 = tf.layers.dense({inputShape: [1024], units: 40, activation: 'sigmoid'});
    const h2 = tf.layers.dense({units: 10, activation: 'softmax'});
    
    // initialize functional model
    const model = tf.sequential({
       layers: [
            h1, 
            h2
        ]
    });
    
    // console log summary info of model
    console.log(model.summary());
    
    // setup sample callback function to happen after fit
    function onEpochEnd(batch, logs) {
        //console.log('Accuracy', logs.acc);
    }


    // async train function so that fit isn't called in parallel 
    async function train() {
    
        // await fit, specifying epochs, batchsize, and 
        await model.fit(data, labels, {
            epochs: 3, 
            batchSize: 1024, 
            callbacks: {onEpochEnd}
        }).then(info => {
            //console.log('Final accuracy', info.history.acc);
            atAcc = info.history.acc;
            // process next iteration of fit

            if(info.history.acc <= .95) {
            train();
            } else {
                console.log('Done');
                progress.finish();
            }
            return;
        })
    }
    
        /*
        OPTIMIZER (sgd, adam)
        LOSS FUNCTION (objective to minimize, measures how wrong. Can use pregiven fns)
        LIST OF METRICS (use 'accuracy' most of the time)
        */
    
        // compile model
        model.compile({
            optimizer: 'sgd',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        })
    
        // manually call train() in setup 
        train();

        return model;
    
    }
