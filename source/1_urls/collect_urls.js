const got = require('got')
const fs = require("fs")

const pathToMemory = __dirname+"/memory.json"

setTimeout(async () => {
    let memory = JSON.parse(fs.readFileSync(pathToMemory))
    
    let iteration = 0
    const numberOfConcurrentFunctions = 20 // the slower the internet, the larger this should be
    ;[...Array(numberOfConcurrentFunctions)].forEach(async () => {
        // just keep grabbing videos
        while (true) {

            // pick a random url
            let urls = Object.keys(memory.urls)
            // log every 100 iterations
            ;(iteration++ % 100 == 0) && console.log(`${urls.length} urls`)
            let randomUrl = urls[Math.floor(Math.random() * urls.length)]
            let html = await getHtmlFor(randomUrl)


            // 
            // extract video id's
            // 
            let videoIds = findAll(
                /href="(?:https:\/\/youtu\.be\/|(?:https:\/\/www\.youtube\.com)?\/watch\?v=)([a-zA-Z0-9_.]{11})(?:"|&)/, html
            ).map(each=>each[1])
            let videoIdMatch = html.match(/videoId:"(.+?)"/)
            if (videoIdMatch) {
                if (videoIdMatch[1].length == 11) {
                    videoIds.push(videoIdMatch[1])
                }
            }

            // 
            // save to file
            //
            let newUrls = videoIds.map(each=>`https://youtu.be/${each}`)
            for (let each of newUrls ) { memory.urls[each]     = {} }
            for (let each of videoIds) { memory.videoIds[each] = {} }
            fs.writeFileSync(pathToMemory, JSON.stringify(memory,0,4))
        }
    })
}, 0)


async function getHtmlFor(url) {
    let html = ""
    try {
        const response = await got(url)
        html = response.body
    } catch (error) {}
    return html
}

function findAll(regexPattern, sourceString) {
    let output = []
    let match
    // make sure the pattern has the global flag
    let regexPatternWithGlobal = RegExp(regexPattern,[...new Set("g"+regexPattern.flags)].join(""))
    while (match = regexPatternWithGlobal.exec(sourceString)) {
        // get rid of the string copy
        delete match.input
        // store the match data
        output.push(match)
    } 
    return output
}