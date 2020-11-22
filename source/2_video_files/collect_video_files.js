const fs = require('fs')
const ytdl = require('ytdl-core')
const util = require('util')
const path = require('path')
const readdir = util.promisify(fs.readdir)
const stat = util.promisify(fs.stat)
const { PATHS } = require("../../toolbox/globals")

const pathToMemory = __dirname+"/memory.json"

// make sure the folder exists
try {
    fs.mkdirSync(PATHS.videoStorage)
} catch (error) {}

let numberOfConcurrentFunctions = 1 // number of simultaneous video downloads
;[...Array(numberOfConcurrentFunctions)].forEach(async () => {

    while (true) {
        try {
            // get all the videoIds
            let videoIds = new Set(Object.keys(JSON.parse(fs.readFileSync(PATHS.videoIdJson)).videoIds))
            let attemptedDownload = new Set(JSON.parse(fs.readFileSync(pathToMemory)).attemptedDownload)
            let videoIdsNotYetAttempted = new Set([...videoIds].filter(each=>!attemptedDownload.has(each)))
            if (videoIdsNotYetAttempted.size <= 0) {
                console.log(`no more video ids avaliable to attempt to download`)
                break
            }
            
            // download all of the them
            while (videoIdsNotYetAttempted.size > 0) {
                // 
                // pick a random id
                // 
                let listOfChoices = [...videoIdsNotYetAttempted]
                let randomId = listOfChoices[Math.floor(Math.random() * listOfChoices.length-0.000001)]
                const sizeOfYoutubeId = 11
                if (typeof randomId == 'string' && randomId.length == sizeOfYoutubeId) {
                    videoIdsNotYetAttempted.delete(randomId)
                    attemptedDownload.add(randomId)
                    
                    // 
                    // try downloading the video
                    // 
                    try {
                        await new Promise((resolve, reject)=>ytdl(randomId).pipe(fs.createWriteStream(`${PATHS.videoStorage}/${randomId}.mp4`)).on('close',resolve).on('error', reject))
                    } catch (error) {
                        console.log(`error downloading ${randomId}`)
                    }
                    fs.writeFileSync(pathToMemory, JSON.stringify({ attemptedDownload: [...attemptedDownload] }))
                    console.log(`downloaded: ${randomId}`)
                }
            }
        } catch (error) {
            console.debug(`error is:`,error)
            console.debug(`error.trace is:`,error.trace)
        }
    }
})
