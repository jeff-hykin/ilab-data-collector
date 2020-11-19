let fs = require("fs")
let path = require("path")

// load up
const INFO       = JSON.parse(fs.readFileSync(__dirname+"/../settings/info.json"))
const PATHS      = processPathsMapping(process.cwd(), INFO["paths"])
const PARAMETERS = INFO["parameters"]
module.exports = {
    INFO,
    PATHS,
    PARAMETERS,
}

function processPathsMapping(source, mapping) {
    let resultingPaths = {}
    for (let [eachKey, eachValue] of Object.entries(mapping)) {
        if (typeof eachKey == 'string') {
            if (typeof eachValue == 'string') {
                // if not absolute
                if (!path.isAbsolute(eachValue)) {
                    // make it absolute
                    resultingPaths[eachKey] = path.join(source, eachValue)
                } else {
                    resultingPaths[eachKey] = eachValue
                }
            }
        }
    }
    return resultingPaths
}